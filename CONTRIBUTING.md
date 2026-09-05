# Contributing to YC2Vec

## The quickest useful contribution

You do not need a GPU, a model, or even network access to work on most of this
project. The fixture profile runs the entire pipeline against a committed
60-company sample using a deterministic in-process model:

```bash
uv sync --all-extras
make fixture          # fetch → discover → review → assign → embed → project → publish → validate
make site-data DATA_DIR=/tmp/yc2vec-fixture
make dev              # the full UI, against the fixture dataset
```

That is the same vertical slice CI runs. If it passes locally it will pass there.

## Before opening a pull request

```bash
make check            # lint, type-check, all tests, fixture pipeline, release gates
```

For frontend changes, also run the browser checks against a real build. They
cover what unit tests cannot see: accessibility basics, deep-link routing under
the GitHub Pages subpath, runtime errors in the WebGL map, and both themes.

```bash
cd frontend
npm run build && npm run preview &
npm run check:browser                          # or: ... -- https://indxdev.github.io/yc2vec/
```

If you have a local Ollama with the configured models, the live-model tests are
worth running before touching a prompt:

```bash
uv run pytest -m ollama
```

## How the pieces fit

| Area | Where | What to know |
| --- | --- | --- |
| Source adapters | `pipeline/adapters/` | Add a source as a new adapter, disabled by default, with a licence review in `DATA_LICENSE.md`. |
| Normalisation | `pipeline/normalize/` | Missing stays missing. Never infer a metadata value. |
| Ontology | `pipeline/ontology/` | Tag ids are minted once and frozen. Merges are migrations, never deletions. |
| Assignment | `pipeline/tagging/` | Every positive needs verifiable evidence. `uncertain` is a real answer. |
| Vectors | `pipeline/embeddings/` | Changing the model or the combination weights creates a new embedding-space version. |
| Publication | `pipeline/publish/` | The browser gets precomputed artifacts only. No backend, ever. |
| Frontend | `frontend/src/` | Similarity comes from precomputed neighbours, never from 2D distance. |

## Rules that are not negotiable

These exist because breaking them silently degrades the dataset in ways that
are hard to notice later.

1. **The published site stays static.** No runtime backend, database, secret or
   model dependency. If a feature needs one, it belongs in the pipeline.
2. **Source classifications are never renamed or merged.** YC's taxonomy and
   YC2Vec's ontology are separate, versioned concepts. The relationship between
   them lives in the mapping table.
3. **No metered model APIs.** Every model call goes through local Ollama.
4. **Untrusted text stays fenced.** Use `wrap_untrusted`; never interpolate
   fetched content into a prompt directly.
5. **Provenance or it did not happen.** A published positive assignment without
   an evidence span and a rationale fails the release gates.
6. **Do not commit raw scraped page bodies.** `data/cache/` is git-ignored for
   this reason.
7. **Bump the version constant when semantics change.** `pipeline/versions.py`
   feeds cache keys; a stage that should have rerun but did not is a silent
   correctness bug.

## Changing a prompt

Prompt text is hashed into cache keys and recorded in every judgment, so editing
one invalidates exactly the derived data it affects. Bump the matching entry in
`PROMPT_VERSIONS` in the same commit, and say in the PR description what you
expect to change about the output.

## Adding a semantic tag by hand

Don't, in general — the ontology is discovered rather than authored, and a
hand-added tag with no discovery support skews the evaluation metrics. If a tag
is genuinely missing, open a tag-review issue with example companies. Real gaps
are usually a discovery or activation-threshold problem, and fixing that helps
every future run.

## Reviewing a data-update pull request

Enrichment PRs contain model inference. Look at `data/quality.json` first:

- `hard_negative_acceptance_rate` is the false-positive proxy. A rise means
  precision fell.
- `evidence_coverage` below 1.0 means positives shipped without provenance —
  the gates should have caught it, so investigate.
- `duplicate_tag_rate` above zero means the merge stage let a duplicate through.
- `orphan_tag_rate` rising means discovery is producing tags the assigner cannot
  ground.

Then read the merge review queue and spot-check a handful of new assignments
against their evidence spans.

## Code style

Python is formatted with `ruff format` and type-checked with `mypy`. TypeScript
is `strict`. Comments should explain why a decision was made, not restate what
the line does; the existing code is the reference.
