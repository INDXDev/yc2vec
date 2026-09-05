"""``yc2vec`` command line interface.

Every command supports ``--help``, bounded concurrency, structured progress
logging and resume. Long stages checkpoint continuously, so interrupting any
command is safe: rerunning it picks up where it stopped.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from pipeline import PIPELINE_VERSION
from pipeline.config import Config, load_config
from pipeline.ollama import ModelNotInstalled, OllamaClient
from pipeline.store import Store
from pipeline.util import log, now, setup_logging, write_json
from pipeline.versions import ONTOLOGY_VERSION, PUBLIC_ARTIFACT_VERSION, SCHEMA_VERSION

app = typer.Typer(
    name="yc2vec",
    help="Build the YC2Vec dataset: fetch, discover tags, assign, embed, project, publish.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
LOG = log("yc2vec")

# Shared option types.
ProfileOpt = Annotated[
    str | None, typer.Option("--profile", "-p", help="fixture | balanced | flagship")
]
DataDirOpt = Annotated[Path | None, typer.Option("--data-dir", help="Override the data directory.")]
ChatModelOpt = Annotated[str | None, typer.Option("--chat-model", help="Ollama chat model to use.")]
EmbedModelOpt = Annotated[
    str | None, typer.Option("--embedding-model", help="Ollama embedding model.")
]
HostOpt = Annotated[str | None, typer.Option("--ollama-host", help="Ollama base URL.")]
LimitOpt = Annotated[int | None, typer.Option("--limit", "-n", help="Process at most N companies.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]
DryRunOpt = Annotated[bool, typer.Option("--dry-run", help="Report what would run, then exit.")]
ForceOpt = Annotated[bool, typer.Option("--force", help="Ignore cached stage keys and rerun.")]


def _ctx(
    profile: str | None,
    data_dir: Path | None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    ollama_host: str | None = None,
    verbose: bool = False,
    enable_crawl: bool | None = None,
) -> tuple[Config, Store]:
    setup_logging(verbose)
    config = load_config(
        profile,
        data_dir=data_dir,
        chat_model=chat_model,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
        enable_crawl=enable_crawl,
    )
    return config, Store(config.data_dir)


def _run(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted -- progress is checkpointed; rerun to resume[/yellow]")
        raise typer.Exit(130) from None
    except ModelNotInstalled as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


async def _client(config: Config, store: Store) -> OllamaClient:
    # The fixture profile uses the deterministic backend, not the offline guard.
    return OllamaClient(config, store, offline=False)


# --------------------------------------------------------------------------


@app.command()
def doctor(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    chat_model: ChatModelOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Check Ollama connectivity, models, disk, RAM and GPU against the profile."""
    config, store = _ctx(profile, data_dir, chat_model, embedding_model, ollama_host, verbose)

    async def go() -> int:
        table = Table(title=f"yc2vec doctor -- profile '{config.profile}'", show_lines=False)
        table.add_column("check")
        table.add_column("result")
        table.add_column("detail", overflow="fold")
        problems = 0

        table.add_row(
            "pipeline",
            "ok",
            f"v{PIPELINE_VERSION}, schema v{SCHEMA_VERSION}, ontology v{ONTOLOGY_VERSION}",
        )
        table.add_row(
            "data dir", "ok" if config.data_dir.exists() else "created", str(config.data_dir)
        )

        du = shutil.disk_usage(config.data_dir)
        free_gb = du.free / 1e9
        ok_disk = free_gb > 20
        problems += not ok_disk
        table.add_row("disk", "ok" if ok_disk else "low", f"{free_gb:.0f} GB free")

        ram = _total_ram_gb()
        table.add_row(
            "ram",
            "ok" if ram is None or ram >= 24 else "low",
            f"{ram:.0f} GB" if ram else "unknown",
        )
        gpu = _gpu_info()
        table.add_row(
            "gpu",
            "ok" if gpu else "none",
            gpu or "no NVIDIA GPU detected (CPU inference will be slow)",
        )

        if config.profile == "fixture":
            table.add_row(
                "ollama", "skipped", "fixture profile uses committed responses; no model required"
            )
            console.print(table)
            return 0

        async with OllamaClient(config, store) as client:
            if not await client.ping():
                table.add_row("ollama", "FAIL", f"cannot reach {config.ollama_host}")
                console.print(table)
                console.print("[red]Start Ollama (`ollama serve`) or set --ollama-host.[/red]")
                return 1
            table.add_row("ollama", "ok", config.ollama_host)
            installed = await client.list_models()
            names = {m.name: m for m in installed}

            for label, wanted in (
                ("chat model", config.models.chat_model),
                ("embedding model", config.models.embedding_model),
            ):
                m = names.get(wanted)
                if m is None:
                    problems += 1
                    table.add_row(
                        label,
                        "FAIL",
                        f"{wanted} not installed. Installed: {', '.join(sorted(names)) or '(none)'}. "
                        f"Run `ollama pull {wanted}` or select an installed model explicitly. "
                        f"YC2Vec never substitutes a model for you.",
                    )
                else:
                    detail = f"{m.size_bytes / 1e9:.1f} GB"
                    if m.parameter_size:
                        detail += f", {m.parameter_size}"
                    if m.quantization:
                        detail += f", {m.quantization}"
                    if m.context_length:
                        detail += f", {m.context_length // 1024}K ctx"
                    detail += f", digest {m.digest[:12]}"
                    table.add_row(label, "ok", detail)
                    # Feasibility: the full quantised weights must be resident.
                    if ram and m.size_bytes / 1e9 > ram * 0.8:
                        problems += 1
                        table.add_row(
                            f"{label} feasibility",
                            "FAIL",
                            f"{m.size_bytes / 1e9:.0f} GB of weights vs {ram:.0f} GB RAM. "
                            "A mixture-of-experts model still needs all quantised weights resident; "
                            "active-parameter count is not the memory requirement.",
                        )

        console.print(table)
        if config.profile == "flagship":
            console.print(
                "[yellow]flagship profile: confirm the model licence permits your use and the "
                "redistribution of derived outputs before running enrichment.[/yellow]"
            )
        return 1 if problems else 0

    code = _run(go())
    if code:
        raise typer.Exit(code)
    console.print("[green]doctor: profile looks feasible.[/green]")


def _total_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        return None
    return None


def _gpu_info() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return " | ".join(out.stdout.strip().splitlines()) or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@app.command()
def fetch(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    limit: LimitOpt = None,
    enable_crawl: Annotated[
        bool, typer.Option("--enable-crawl/--no-crawl", help="Opt in to website enrichment.")
    ] = False,
    changed_since: Annotated[
        bool, typer.Option("--changed-since/--full", help="Reuse unchanged records.")
    ] = True,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Fetch structured company records and the YC source taxonomy, then normalise."""
    config, store = _ctx(profile, data_dir, verbose=verbose, enable_crawl=enable_crawl)
    if dry_run:
        console.print(
            f"would fetch {config.source_base_url}/companies/all.json"
            f"{f' (limit {limit})' if limit else ''}, crawl={'on' if config.crawl.enabled else 'off'}"
        )
        return

    from pipeline.fetch.stage import fetch_stage

    prior = store.cache_get("source", "raw_hashes") if changed_since else None
    counts = _run(fetch_stage(config, store, limit=limit, changed_since_hashes=prior))
    store.record(
        "fetch",
        store.stage_key("fetch", {"limit": limit, "url": config.source_base_url}),
        [store.path("normalized", "companies.jsonl")],
        counts,
    )
    console.print(counts)


@app.command("discover-tags")
def discover_tags(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    chat_model: ChatModelOpt = None,
    ollama_host: HostOpt = None,
    limit: LimitOpt = None,
    max_batches: Annotated[
        int | None, typer.Option("--max-batches", help="Cap discovery batches.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Concurrent discovery calls.")
    ] = 4,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Ask the local model for reusable semantic attributes over diverse company batches."""
    config, store = _ctx(profile, data_dir, chat_model, None, ollama_host, verbose)
    if dry_run:
        console.print(
            f"would run discovery with {config.models.chat_model}, batch size "
            f"{config.ontology.discovery_batch_size}, max_batches={max_batches}"
        )
        return

    from pipeline.orchestrate import discover_stage

    async def go() -> dict[str, int]:
        async with await _client(config, store) as client:
            return await discover_stage(
                config, store, client, max_batches=max_batches, limit=limit, concurrency=concurrency
            )

    console.print(_run(go()))


@app.command("review-tags")
def review_tags(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    chat_model: ChatModelOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    apply: Annotated[
        bool, typer.Option("--apply/--propose-only", help="Apply approved merges.")
    ] = True,
    max_adjudications: Annotated[
        int,
        typer.Option(
            "--max-adjudications",
            help="Cap LLM merge adjudications; the rest stay queued for review.",
        ),
    ] = 400,
    min_support: Annotated[
        int | None, typer.Option("--min-support", help="Override the activation support threshold.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Concurrent adjudication calls.")
    ] = 4,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Normalise, adjudicate merges and activate candidate tags."""
    config, store = _ctx(profile, data_dir, chat_model, embedding_model, ollama_host, verbose)
    if dry_run:
        console.print(
            f"would review the ontology (auto-merge >= {config.ontology.auto_merge_threshold}, "
            f"review >= {config.ontology.review_threshold}, min support {config.ontology.min_support})"
        )
        return

    from pipeline.orchestrate import review_stage

    async def go() -> dict[str, int]:
        async with await _client(config, store) as client:
            cfg = config
            if min_support is not None:
                from dataclasses import replace as _replace

                cfg = _replace(config, ontology=_replace(config.ontology, min_support=min_support))
            return await review_stage(
                cfg,
                store,
                client,
                apply=apply,
                max_adjudications=max_adjudications,
                concurrency=concurrency,
            )

    console.print(_run(go()))


@app.command("map-taxonomy")
def map_taxonomy(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Relate YC source taxonomy terms to YC2Vec semantic tags (comparison only)."""
    config, store = _ctx(profile, data_dir, None, embedding_model, ollama_host, verbose)
    from pipeline.orchestrate import map_taxonomy_stage

    async def go() -> dict[str, int]:
        async with await _client(config, store) as client:
            return await map_taxonomy_stage(config, store, client)

    console.print(_run(go()))


@app.command("assign-tags")
def assign_tags(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    chat_model: ChatModelOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    limit: LimitOpt = None,
    companies: Annotated[
        str | None, typer.Option("--companies", help="Comma-separated company ids.")
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option(
            "--sample",
            help="Judge a representative sample of N companies, stratified by industry and batch year.",
        ),
    ] = None,
    shortlist: Annotated[
        int | None,
        typer.Option("--shortlist", help="Tags shortlisted per company before hard negatives."),
    ] = None,
    hard_negatives: Annotated[
        int | None,
        typer.Option("--hard-negatives", help="Calibrated hard negatives added per company."),
    ] = None,
    pairs_per_call: Annotated[
        int | None,
        typer.Option(
            "--pairs-per-call", help="Pairs judged per model call. 1 = strictly one call per pair."
        ),
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="Concurrent companies in flight.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--restart", help="Reuse checkpointed judgments.")
    ] = True,
    finalize: Annotated[
        bool,
        typer.Option(
            "--finalize",
            help="Consolidate the existing checkpoint into the final tables without judging anything new.",
        ),
    ] = False,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Judge shortlisted company/tag pairs with evidence, then build sparse features."""
    from dataclasses import replace as _replace

    config, store = _ctx(profile, data_dir, chat_model, embedding_model, ollama_host, verbose)
    ids = [c.strip() for c in companies.split(",") if c.strip()] if companies else None

    overrides = {
        k: v
        for k, v in {
            "shortlist_size": shortlist,
            "hard_negatives": hard_negatives,
            "pairs_per_call": pairs_per_call,
            "concurrency": concurrency,
        }.items()
        if v is not None
    }
    cfg = _replace(config, tagging=_replace(config.tagging, **overrides)) if overrides else config

    # Report the settings that would actually be used, overrides included.
    if dry_run:
        console.print(
            f"would judge up to {cfg.tagging.shortlist_size} tags + "
            f"{cfg.tagging.hard_negatives} hard negatives per company, "
            f"{cfg.tagging.pairs_per_call} pairs per call, concurrency {cfg.tagging.concurrency}"
            + (f", over a stratified sample of {sample}" if sample else "")
        )
        return

    from pipeline.orchestrate import assign_stage

    async def go() -> dict[str, int]:
        async with await _client(cfg, store) as client:
            return await assign_stage(
                cfg,
                store,
                client,
                limit=limit,
                company_ids=ids,
                sample=sample,
                resume=resume,
                finalize_only=finalize,
            )

    console.print(_run(go()))


@app.command()
def embed(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    limit: LimitOpt = None,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Build description, metadata, tag and combined vectors, then top-K neighbours."""
    config, store = _ctx(profile, data_dir, None, embedding_model, ollama_host, verbose)
    if dry_run:
        console.print(
            f"would embed 4 documents per company with {config.models.embedding_model} "
            f"and compute top-{config.embeddings.top_k_neighbors} neighbours in 5 spaces"
        )
        return

    from pipeline.orchestrate import embed_stage

    async def go() -> dict[str, int]:
        async with await _client(config, store) as client:
            return await embed_stage(config, store, client, limit=limit)

    console.print(_run(go()))


@app.command()
def project(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    align: Annotated[
        bool, typer.Option("--align/--no-align", help="Align to the previous release.")
    ] = True,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Fit the 2D UMAP projection and label algorithmic clusters."""
    config, store = _ctx(profile, data_dir, verbose=verbose)
    if dry_run:
        console.print(
            f"would fit UMAP (n_neighbors={config.projection.n_neighbors}, "
            f"min_dist={config.projection.min_dist}, seed={config.projection.seed})"
        )
        return

    from pipeline.orchestrate import project_stage

    console.print(project_stage(config, store, align=align))


@app.command("publish-data")
def publish_data(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    skip_exports: Annotated[
        bool, typer.Option("--skip-exports", help="Browser artifacts only.")
    ] = False,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Write browser artifacts, CSV/Parquet exports and the release manifest."""
    config, store = _ctx(profile, data_dir, verbose=verbose)
    if dry_run:
        console.print(f"would publish to {config.public_dir} and export to {config.export_dir}")
        return

    from pipeline.fetch.stage import source_meta
    from pipeline.models import ReleaseManifest
    from pipeline.orchestrate import dataset_version, load_artifacts, load_embeddings
    from pipeline.prompts import prompt_hashes
    from pipeline.publish.browser import publish_browser_artifacts
    from pipeline.publish.exports import write_exports
    from pipeline.quality.evaluation import evaluate_dataset
    from pipeline.util import git_commit

    a = load_artifacts(store)
    src = source_meta(store)
    space = a["neighbors"][0].embedding_space_version if a["neighbors"] else "emb-none"
    projection = a["points"][0].projection_version if a["points"] else "umap-none"

    manifest = ReleaseManifest(
        dataset_version=dataset_version(store, a),
        schema_version=SCHEMA_VERSION,
        public_artifact_version=PUBLIC_ARTIFACT_VERSION,
        pipeline_version=PIPELINE_VERSION,
        ontology_version=ONTOLOGY_VERSION,
        embedding_space_version=space,
        projection_version=projection,
        generated_at=now(),
        git_commit=git_commit(),
        source_retrieved_at=_parse_iso(src.get("retrieved_at")),
        source_last_updated=src.get("source_last_updated") or None,
        source_url=src.get("source_url") or config.source_base_url,
        models={
            "chat": config.models.chat_model,
            "embedding": config.models.embedding_model,
            "profile": config.profile,
            "temperature": config.models.temperature,
            "seed": config.models.seed,
        },
        prompt_hashes=prompt_hashes(),
        limitations=[
            "Semantic tags are inferred by a local language model and are not verified by "
            "Y Combinator or by the companies described.",
            "UMAP coordinates are a lossy 2D projection; use the precomputed neighbour lists "
            "for similarity, not chart distance.",
            "Clusters are algorithmic groupings, not official categories.",
            "Coverage depends on what the public source records contain; missing metadata is "
            "left missing rather than inferred.",
        ],
        licenses=[
            {"name": "yc-oss/api", "url": "https://github.com/yc-oss/api", "role": "source data"},
            {"name": "YC2Vec code", "url": "https://github.com/INDXDev/yc2vec", "spdx": "MIT"},
            {"name": "YC2Vec derived data", "spdx": "CC-BY-4.0", "role": "derived dataset"},
        ],
        attribution=(
            "Company records derive from the open-source yc-oss/api project, which republishes "
            "Y Combinator's public company index. YC2Vec is an independent, unofficial project "
            "and is not affiliated with or endorsed by Y Combinator."
        ),
    )

    active = [t for t in a["tags"] if t.state == "active"]
    counts = publish_browser_artifacts(
        config.public_dir,
        companies=a["companies"],
        tags=active,
        features=a["features"],
        judgments=a["judgments"],
        neighbors=a["neighbors"],
        points=a["points"],
        clusters=a["clusters"],
        terms=a["terms"],
        mappings=a["mappings"],
        manifest=manifest,
    )
    if not skip_exports:
        counts.update(
            write_exports(
                config.export_dir,
                companies=a["companies"],
                tags=a["tags"],
                terms=a["terms"],
                mappings=a["mappings"],
                features=a["features"],
                judgments=a["judgments"],
                neighbors=a["neighbors"],
                points=a["points"],
                embeddings=None,
            )
        )
        _ = load_embeddings

    from pipeline.models import CompanyTagJudgment
    from pipeline.util import read_jsonl

    checkpoint = [
        CompanyTagJudgment(**r)
        for r in read_jsonl(store.path("inferred", "company_tag_judgments.partial.jsonl"))
    ]
    metrics = evaluate_dataset(
        companies_count=len(a["companies"]),
        tags=a["tags"],
        features=a["features"],
        judgments=a["judgments"],
        checkpoint_judgments=checkpoint or None,
        gold_path=Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "gold"
        / "judgments.json",
    )
    write_json(
        config.public_dir / f"v{PUBLIC_ARTIFACT_VERSION}" / "quality.json", metrics, pretty=True
    )
    write_json(config.data_dir / "quality.json", metrics, pretty=True)
    console.print(
        {
            **counts,
            "quality": {
                k: metrics[k]
                for k in ("active_tags", "assignments", "evidence_coverage", "uncertain_rate")
            },
        }
    )


@app.command()
def validate(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    published_only: Annotated[
        bool,
        typer.Option(
            "--published-only",
            help="Check only the published artifacts. Use where the intermediate tables are not available, such as a deploy job.",
        ),
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run every release gate. Exits non-zero on any failure."""
    config, store = _ctx(profile, data_dir, verbose=verbose)

    if published_only:
        from pipeline.quality.published import run_published_gates

        results = run_published_gates(config.public_dir)
        _print_gates(results)
        if any(not r.passed for r in results):
            raise typer.Exit(1)
        return

    from pipeline.orchestrate import load_artifacts
    from pipeline.quality.gates import run_release_gates

    a = load_artifacts(store)
    results = run_release_gates(
        companies=a["companies"],
        tags=a["tags"],
        features=a["features"],
        judgments=a["judgments"],
        neighbors=a["neighbors"],
        points=a["points"],
        public_dir=config.public_dir,
    )
    _print_gates(results)
    if any(not r.passed for r in results):
        raise typer.Exit(1)


def _print_gates(results: list[Any]) -> None:
    table = Table(title="release gates")
    table.add_column("gate")
    table.add_column("result")
    table.add_column("detail", overflow="fold")
    for r in results:
        table.add_row(
            r.name,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            r.detail + (f" e.g. {', '.join(r.samples[:3])}" if r.samples else ""),
        )
    console.print(table)


@app.command("migrate-ontology")
def migrate_ontology(
    migration: Annotated[
        str, typer.Argument(help="Migration to run. Use 'list' to see what is available.")
    ] = "list",
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Apply an ontology migration. Tag ids are never changed."""
    config, store = _ctx(profile, data_dir, verbose=verbose)
    from pipeline.ontology.migrations import MIGRATIONS
    from pipeline.ontology.registry import OntologyRegistry

    if migration == "list" or migration not in MIGRATIONS:
        if migration != "list":
            console.print(f"[red]unknown migration {migration!r}[/red]")
        console.print("available migrations:")
        for name, fn in MIGRATIONS.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            console.print(f"  {name}  —  {summary}")
        raise typer.Exit(0 if migration == "list" else 2)

    registry = OntologyRegistry(store.path("inferred", "ontology"))
    result = MIGRATIONS[migration](registry, dry_run=dry_run)
    if not dry_run and result.changed:
        registry.save()
    console.print(
        {
            "migration": result.name,
            "changed": result.changed,
            "applied": not dry_run,
            "examples": result.examples,
        }
    )
    _ = config


@app.command()
def schemas(
    out: Annotated[Path, typer.Option("--out", help="Where to write JSON Schema files.")] = Path(
        "pipeline/schemas"
    ),
) -> None:
    """Emit JSON Schema for every data contract (shared with the frontend types)."""
    from pipeline.models import ALL_MODELS

    out.mkdir(parents=True, exist_ok=True)
    for name, model in ALL_MODELS.items():
        write_json(out / f"{name}.schema.json", model.model_json_schema(), pretty=True)
    console.print(f"wrote {len(ALL_MODELS)} schemas to {out}")


@app.command()
def stats(profile: ProfileOpt = None, data_dir: DataDirOpt = None) -> None:
    """Print dataset counts and quality metrics."""
    config, store = _ctx(profile, data_dir)
    quality = config.data_dir / "quality.json"
    if quality.exists():
        console.print_json(quality.read_text())
    else:
        console.print(json.dumps(store._manifest.get("stages", {}), indent=2))


@app.command()
def run(
    profile: ProfileOpt = None,
    data_dir: DataDirOpt = None,
    chat_model: ChatModelOpt = None,
    embedding_model: EmbedModelOpt = None,
    ollama_host: HostOpt = None,
    limit: LimitOpt = None,
    incremental: Annotated[
        bool, typer.Option("--incremental/--full", help="Skip stages whose inputs are unchanged.")
    ] = True,
    force_stage: Annotated[
        str | None, typer.Option("--force-stage", help="Comma-separated stages to rerun.")
    ] = None,
    skip: Annotated[
        str | None, typer.Option("--skip", help="Comma-separated stages to skip.")
    ] = None,
    enable_crawl: Annotated[bool, typer.Option("--enable-crawl/--no-crawl")] = False,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run the whole pipeline end to end, reusing fresh stages."""
    config, store = _ctx(
        profile, data_dir, chat_model, embedding_model, ollama_host, verbose, enable_crawl
    )
    forced = {s.strip() for s in (force_stage or "").split(",") if s.strip()}
    skipped = {s.strip() for s in (skip or "").split(",") if s.strip()}
    stages = [
        "fetch",
        "discover-tags",
        "review-tags",
        "map-taxonomy",
        "assign-tags",
        "embed",
        "project",
        "publish-data",
    ]
    if forced:
        store.invalidate(sorted(forced))

    if dry_run:
        console.print(f"would run: {', '.join(s for s in stages if s not in skipped)}")
        console.print(
            f"forced={sorted(forced) or '-'} skipped={sorted(skipped) or '-'} incremental={incremental}"
        )
        return

    from pipeline.fetch.stage import fetch_stage
    from pipeline.orchestrate import (
        assign_stage,
        discover_stage,
        embed_stage,
        map_taxonomy_stage,
        project_stage,
        review_stage,
    )

    async def go() -> None:
        async with await _client(config, store) as client:
            for stage in stages:
                if stage in skipped:
                    LOG.info("skipping %s", stage)
                    continue
                key = store.stage_key(stage, {"config": config.fingerprint(), "limit": limit})
                if incremental and stage not in forced and store.is_fresh(stage, key):
                    LOG.info("%s is fresh; skipping", stage)
                    continue
                LOG.info("=== %s ===", stage)
                if stage == "fetch":
                    counts = await fetch_stage(
                        config,
                        store,
                        limit=limit,
                        changed_since_hashes=store.cache_get("source", "raw_hashes"),
                    )
                elif stage == "discover-tags":
                    counts = await discover_stage(config, store, client, limit=limit)
                elif stage == "review-tags":
                    counts = await review_stage(config, store, client)
                elif stage == "map-taxonomy":
                    counts = await map_taxonomy_stage(config, store, client)
                elif stage == "assign-tags":
                    counts = await assign_stage(config, store, client, limit=limit)
                elif stage == "embed":
                    counts = await embed_stage(config, store, client, limit=limit)
                elif stage == "project":
                    counts = project_stage(config, store)
                else:
                    counts = {}
                store.record(stage, key, counts=counts)

    _run(go())
    if "publish-data" not in skipped:
        publish_data(profile=profile, data_dir=data_dir, verbose=verbose)
        validate(profile=profile, data_dir=data_dir, verbose=verbose)


def _parse_iso(value: str | None) -> datetime | None:
    """Source timestamps round-trip through JSON as strings."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
