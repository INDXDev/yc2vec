# YC2Vec

**An open-data map of the semantic DNA of Y Combinator companies, built entirely with local models.**

🗺️ **[Explore the map →](https://indxdev.github.io/yc2vec/)**

> **Unofficial project.** YC2Vec is an independent open-data experiment. It is not
> affiliated with, endorsed by, or connected to Y Combinator. Company records come
> from the open-source [yc-oss/api](https://github.com/yc-oss/api) project, which
> republishes YC's public company index. Semantic tags are inferred by a language
> model and are not verified by Y Combinator or by the companies described.

---

## What this is

YC's own directory tells you a company is in "B2B" or "Fintech". That is true and
almost never the interesting part. YC2Vec asks a different question: if you
described every company by *what it actually does* — who buys it, what workflow it
replaces, what data it operates on, how it is delivered — what shape would the
ecosystem have?

To answer that, a local model reads every company's public description, proposes a
reusable vocabulary of semantic attributes, curates that vocabulary down to a
deduplicated ontology, and then judges each company against the attributes that
plausibly apply — with evidence, a rationale, and the right to say "I don't know".

Each company ends up with three linked representations:

1. a **sparse, interpretable** company × tag vector you can read and audit;
2. **dense embeddings** of its description, its metadata, its tag profile, and a
   documented combination of the three;
3. a **2D UMAP projection** for looking at.

The site makes the point of vectorisation obvious: you can see clusters, filter by
metadata and by inferred tags, ask *why* a tag was assigned and read the sentence
that justified it, and find the most similar companies in five different senses of
"similar".

**Zero token cost.** Every model call runs locally through Ollama. There are no
metered API fees. That is not the same as free — it costs hardware, electricity,
bandwidth and time, and the full quantised weights must stay resident in memory.

---

## Try it without a GPU

The fixture profile runs the **entire pipeline** — discovery, merge review,
assignment, embeddings, neighbours, projection, publication and the release gates —
against a committed 60-company sample, using a deterministic in-process model. No
download, no network, about fifteen seconds.

```bash
git clone https://github.com/INDXDev/yc2vec && cd yc2vec
uv sync --all-extras
make fixture                                  # the whole pipeline, no model needed
make site-data DATA_DIR=/tmp/yc2vec-fixture   # stage the result for the UI
cd frontend && npm ci && npm run dev
```

That is the same vertical slice CI runs on every pull request.

---

## Run it for real

### Hardware

| Profile | Models | Needs | Notes |
| --- | --- | --- | --- |
| `fixture` | none | nothing | Committed sample + deterministic stub. Used by CI. |
| `small` | `qwen3:8b`, `qwen3-embedding:0.6b` | ~12 GB | Works. Coarser ontology, noisier assignments. |
| `balanced` *(default)* | `qwen3.8:27b`, `qwen3-embedding:8b` | ~24 GB, GPU strongly preferred | What the published dataset was built with. |
| `flagship` | an explicitly selected large local model | see below | Opt-in only. |

`yc2vec doctor` inspects Ollama, the installed models, disk, RAM and GPU, and tells
you whether the profile is feasible. It will never quietly substitute a different
model — if the configured one is missing it says so and lists what you have.

For `flagship`: a mixture-of-experts model still requires the **full quantised
weights** resident across RAM and VRAM. The active-parameter count is not the
memory requirement. Check the model's licence permits your use and the
redistribution of derived outputs before running enrichment.

### Setup

```bash
# Ollama on the host (or `docker compose --profile ollama up -d`)
ollama pull qwen3.8:27b
ollama pull qwen3-embedding:8b

cp .env.example .env      # every default is sensible; no secrets required
make doctor
```

**Throughput matters more than you would guess.** Ollama serialises concurrent
requests unless you tell it otherwise, which makes the pipeline's concurrency
settings do nothing. Start the server with parallel decoding enabled:

```bash
OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=4h ollama serve
```

Generation throughput is the binding constraint on a full run: the assignment stage
is dominated by output tokens, so aggregate tokens/second is the number to watch.

### The pipeline

```bash
uv run yc2vec doctor          # check models and hardware against the profile
uv run yc2vec fetch           # public company records + YC's own taxonomy
uv run yc2vec discover-tags   # propose reusable semantic attributes
uv run yc2vec review-tags     # normalise, adjudicate merges, activate
uv run yc2vec map-taxonomy    # relate YC's categories to the inferred tags
uv run yc2vec assign-tags     # judge shortlisted pairs, with evidence
uv run yc2vec embed           # four vector spaces + precomputed neighbours
uv run yc2vec project         # UMAP + algorithmic clusters
uv run yc2vec publish-data    # browser artifacts, CSV, Parquet, manifest
uv run yc2vec validate        # release gates

uv run yc2vec run --incremental   # all of the above, skipping fresh stages
```

Every command takes `--help`, `--dry-run` where meaningful, `--limit`, `--profile`,
`--data-dir`, explicit model overrides, and `--verbose`. Everything is **resumable**:
long stages checkpoint after each company, so interrupting any command is safe —
rerun it and it continues.

Useful selectors:

```bash
uv run yc2vec assign-tags --companies ycoss:5,ycoss:86   # specific companies
uv run yc2vec run --force-stage embed,project            # force a rerun
uv run yc2vec run --skip discover-tags                   # skip a stage
uv run yc2vec discover-tags --max-batches 50 --concurrency 6
```

### Website enrichment (optional, off by default)

```bash
uv run yc2vec fetch --enable-crawl
```

Read [DATA_LICENSE.md](DATA_LICENSE.md) first. When enabled, the crawler obeys
`robots.txt`, identifies itself with a contact URL, throttles per domain, caps
response size and content type, restricts and re-validates every redirect, refuses
private and cloud-metadata addresses, and never executes page JavaScript. Only
extracted main text is stored, and only in the git-ignored local cache — raw page
bodies are never committed.

---

## How incrementality works

Each stage declares its inputs — upstream artifact hashes, the relevant config
slice, prompt text, model identity, pipeline version — and the store hashes that
into a **stage key**. A stage whose key is unchanged is skipped.

| Change | What reruns |
| --- | --- |
| A new company appears | Normalisation for it, then the indexes it touches |
| A tag is activated | Shortlisting and judging for plausible companies plus a negative sample |
| Two tags merge | An explicit migration, then the affected sparse vectors |
| A prompt is edited | Exactly the stage that uses that prompt |
| The embedding model changes | A new embedding-space version: vectors, neighbours, projection |

Writes are atomic, so an interrupted publish never corrupts the last good release.

---

## Data outputs

`data/export/`

| File | Contents |
| --- | --- |
| `companies.csv` / `.parquet` | One row per company with normalised metadata |
| `tags.csv` / `.parquet` | The ontology: definitions, facets, aliases, lifecycle, provenance |
| `source_taxonomy.csv` | YC's own industries, tags and batches, reproduced exactly |
| `source_taxonomy_tag_mappings.csv` | How YC's terms relate to the inferred tags |
| `company_tags.csv` | **Long-form** sparse assignments with confidence and provenance |
| `company_tag_matrix.npz` | CSR matrix plus row/column maps, for research use |
| `company_tag_judgments.parquet` | Every judgment including `no` and `uncertain`, as audit data |
| `company_neighbors.parquet` | Precomputed top-K in all five similarity spaces |
| `umap_points.csv` / `.parquet` | 2D coordinates and cluster assignment |

The company × tag relationship is exported long-form on purpose. A 1,024-column CSV
does not scale with an open-ended ontology and hides the confidence and provenance
that make an assignment auditable.

`data/public/v1/` holds the browser-ready artifacts the site serves. Every release
carries a manifest with schema and dataset versions, source timestamps, counts,
model identifiers, prompt hashes, the git commit, per-file checksums, generation
time, known limitations and licences.

---

## Quality

`data/quality.json`, surfaced on the About page. The numbers that matter:

- **`hard_negative_acceptance_rate`** — every company is judged against a few
  deliberately plausible-but-probably-wrong tags. The share the model accepts is
  the honest estimate of its false-positive rate. It is published, not hidden.
- **`evidence_coverage`** — fraction of positives with a verified evidence span.
  A positive whose quote does not literally occur in the evidence is discarded, and
  one left without any evidence is downgraded to `uncertain`.
- **`uncertain_rate`** — how often the model declined to decide. A low number here
  is suspicious, not reassuring.
- **`duplicate_tag_rate`** — should be zero; anything else means the merge stage
  let a duplicate through.
- **`orphan_tag_rate`** — activated tags never actually assigned.
- **`reviewed_sample`** — precision and recall against the human-reviewed gold set
  in `tests/fixtures/gold/`.

### Release gates

`yc2vec validate` blocks a release on any of:

- a schema validation failure;
- a published positive assignment without evidence or a rationale;
- any NaN or infinity;
- a company without a stable id and source URL;
- a neighbour that does not exist, or neighbours spanning more than one
  embedding-space version;
- an assignment referencing an inactive tag or a missing company;
- a public artifact whose checksum does not match the manifest.

---

## Deploying to GitHub Pages

Two manual steps, once:

1. **Settings → Pages → Source: GitHub Actions.**
2. Push to `main`. The `Deploy Pages` workflow validates the dataset, builds the
   frontend with the correct base path, and deploys through the official Pages
   actions.

For a custom domain, set the repository variable `CUSTOM_DOMAIN=1` so the build
uses `/` instead of `/<repo>/`, and add your `CNAME` in the Pages settings.

The app uses hash routing. GitHub Pages has no rewrite rules, so a history route
like `/yc2vec/company/ycoss:5` would 404 on a direct refresh; hash routes are served
by `index.html` at every depth and work identically under a project subpath and on a
custom domain.

### Enabling the self-hosted enrichment runner

The model-dependent stages need a machine with Ollama, so they run on a labelled
self-hosted runner and are triggered manually:

1. **Settings → Actions → Runners → New self-hosted runner**, and give it the
   labels `self-hosted` and `ollama`.
2. On that machine: install Ollama, `ollama pull` both configured models, and
   verify with `uv run yc2vec doctor`.
3. Keep the Ollama port bound to loopback. **Do not expose it publicly** — it has
   no authentication.
4. Run the **Semantic enrichment** workflow from the Actions tab. It checkpoints,
   uploads logs and the quality report, and opens a PR with the derived data.

Enrichment output is model inference and always lands in a pull request for review.
Ontology merges and low-confidence assignments are never auto-committed to `main`.

---

## Repository layout

```
pipeline/
  adapters/      source adapters + URL safety (SSRF defence)
  fetch/         the fetch/normalise stage
  normalize/     field cleanup, ids, the deterministic metadata document
  ontology/      registry, discovery, merge review, taxonomy mapping
  tagging/       pair shortlisting, evidence-grounded judgment, features
  embeddings/    the four vector spaces, combination, neighbours
  projection/    UMAP, projection alignment, cluster labelling
  publish/       browser artifacts, CSV/Parquet/sparse exports
  quality/       release gates and evaluation metrics
  cli.py         the yc2vec command
frontend/src/
  components/    scatterplot, filters, search box, preview
  views/         DNA map, company, tag explorer, search explorer, about
  lib/           data loading, search, filter + URL state, types
tests/
  unit/          pipeline tests (no model or network required)
  fixtures/      committed sample corpus, gold review set
config/          small / medium / flagship profiles
docs/            architecture notes
```

[docs/architecture.md](docs/architecture.md) explains why the stages are split the
way they are and what would be easy to get wrong.
[docs/methodology.md](docs/methodology.md) is the long form of the About page:
what the numbers measure and what they do not support.

---

## Limitations

- **Semantic tags are model inference.** They are not verified by YC or by the
  companies described. Every one carries its evidence so you can check it.
- **Tag assignment is incremental.** A company with no tags has not been processed
  yet, or its public text was too thin to ground any attribute. That is not a
  statement about the company.
- **UMAP is lossy.** Chart distance does not faithfully preserve high-dimensional
  similarity. Every similarity claim in the UI comes from precomputed neighbours in
  the full space, never from the 2D layout.
- **Clusters are algorithmic**, labelled from over-represented tags. They are not
  official categories.
- **Coverage is bounded by the source.** Missing metadata stays missing; the model
  is never asked to fill it in.
- **Generation is not bit-reproducible.** Temperature is zero and every seed is
  fixed, but output can still vary across hardware and runtime versions. The
  deterministic parts — normalisation, vector arithmetic, UMAP with a fixed seed,
  publication — are reproducible.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). `make check` runs what CI runs. There are
issue templates for [data corrections](.github/ISSUE_TEMPLATE/data-correction.yml)
and [tag review](.github/ISSUE_TEMPLATE/tag-review.yml).

## Licences

- **Code:** MIT ([LICENSE](LICENSE))
- **Derived dataset:** CC BY 4.0
- **Source data:** from [yc-oss/api](https://github.com/yc-oss/api) (MIT). Company
  names, logos and descriptions remain the property of the companies described.

See [DATA_LICENSE.md](DATA_LICENSE.md) for the full source review, redistribution
policy and opt-out process. Security policy: [SECURITY.md](SECURITY.md).
