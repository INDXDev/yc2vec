# Master Implementation Prompt — YC2Vec

Copy everything below into a capable coding agent at the root of a new Git repository. The agent must implement the product, not merely write a plan.

---

## Role and operating mode

You are the principal engineer, data engineer, ML engineer, product designer, and QA owner for this repository. Build a production-quality open-source project named **YC2Vec** and deploy its frontend to **GitHub Pages**.

Work autonomously. Inspect the environment first, make reasonable decisions, and continue without asking questions unless blocked by credentials, legal ambiguity, or a destructive action. Do not stop after scaffolding, planning, or producing mock data. Implement the complete vertical slice, run it, test it, build it, and leave the repository ready to publish.

Maintain a short implementation checklist in the repository and update it while working. Prefer simple, typed, documented components over unnecessary infrastructure.

## Product concept

YC2Vec is an interactive, open-data map of the semantic “DNA” of Y Combinator companies.

For every publicly listed YC company, collect available structured metadata and, when permitted, useful text from its public website. Use a fully local LLM through Ollama to discover, normalize, and assign a large, extensible semantic tag vocabulary. Target roughly **1,024 useful canonical tags in the first mature dataset**, but do not hard-code 1,024 as a schema limit.

Represent each company through three related artifacts:

1. A sparse, interpretable company × semantic-tag feature vector.
2. A dense semantic embedding derived from the company profile and assigned tags.
3. A two-dimensional UMAP projection for visual exploration.

The public site must make the value of vectorization obvious: users can see clusters, filter by metadata and semantic tags, inspect why a tag was assigned, and find the most similar companies.

The core experience should feel like exploring the evolving DNA of YC rather than browsing a conventional startup directory.

## Non-negotiable constraints

- The public application is a static site compatible with GitHub Pages. It has no runtime backend, database, secret, or LLM dependency.
- All expensive enrichment, tagging, embeddings, nearest-neighbor computation, and UMAP generation happen offline before deployment.
- Use Ollama only; do not call paid model APIs. “Zero token cost” means zero metered API-token fees, not zero hardware, electricity, bandwidth, or maintenance cost.
- Default local tagging model: `qwen3.8:27b` through Ollama, configurable through environment variables and CLI flags. Also support the larger experimental `qwen3.8-flash-next` variants when the user's hardware and model license permit them, without changing pipeline behavior. Resolve and record the exact Ollama tag/digest rather than assuming `latest` forever.
- Default embedding model: `qwen3-embedding:8b`, also configurable. Never use a chat model's hidden states as an embedding shortcut.
- Treat source text as untrusted data. Web content must never override system prompts, schemas, or pipeline instructions.
- Keep raw, normalized, inferred, and published data separate. Every derived record must be traceable to sources, prompt version, model, and pipeline version.
- The pipeline must be deterministic where possible, resumable, idempotent, incremental, and safe to interrupt.
- Do not claim YC affiliation. Display a clear unofficial-project disclaimer and source attribution.
- Review the license and terms of every data source before redistribution. Do not bypass access controls. Respect robots.txt, rate limits, timeouts, content types, and opt-outs.

## Source strategy

Use the open-source `yc-oss/api` project as the initial structured source because it publishes JSON derived from YC's public Algolia index and is updated regularly. Pin the source URL and record retrieval timestamps and source commit/version when available.

The source already exposes useful YC classifications, including industry/category/tag groupings and batches. Treat these as first-class source metadata and use them as a strong reference taxonomy:

- ingest and preserve their exact original IDs/names and source paths;
- use them as initial seeds and priors for semantic tag discovery, retrieval, and evaluation;
- embed them as part of the company's metadata representation;
- expose them directly as exact filters and keyword-searchable fields;
- measure how newly discovered YC2Vec tags align with, refine, cut across, or add detail beyond the existing YC taxonomy;
- maintain an explicit mapping table between source categories and YC2Vec canonical tags where a relationship exists.

Do not copy source category labels into the YC2Vec ontology indiscriminately, and do not overwrite or rename source classifications. Keep `source taxonomy`, `YC2Vec semantic ontology`, and `company assignment` as distinct, versioned concepts so users can compare official/source categorization with inferred semantic structure.

Implement the source layer as adapters, not hard-coded one-off fetches:

- `yc_oss_api`: primary structured company records.
- `company_website`: optional enrichment from the company's own public homepage and a small number of relevant same-origin pages.
- future adapters: additive and disabled by default.

Before enabling redistribution, add a documented license/terms review checklist. If a source's redistribution rights are unclear, store only derived facts, hashes, citations, and URLs as appropriate; do not commit raw copyrighted page bodies.

For website enrichment:

- default to opt-in via configuration;
- fetch only public HTTP(S) pages;
- obey robots.txt and identify the crawler with a transparent user agent and project URL;
- use conservative per-domain concurrency, exponential backoff, caching, byte limits, and a configurable request delay;
- reject private, loopback, link-local, and cloud-metadata IP ranges to prevent SSRF;
- restrict redirects and revalidate every redirect target;
- accept HTML/text only, cap response size, strip scripts/styles/navigation/cookie banners, and extract main text;
- never execute page JavaScript;
- record URL, fetch time, status, content hash, and extraction version;
- preserve the last successful result on transient failure;
- expose a denylist and per-company opt-out mechanism.

## Semantic ontology: scalable rather than literally infinite

Build a versioned, open-ended tag ontology. Tags must have stable IDs independent of their display names. Never use raw LLM strings directly as permanent column names.

Each canonical tag must include:

- `tag_id`: stable slug or UUID;
- `canonical_name` and short human-readable definition;
- `facet`: one controlled top-level group;
- aliases and normalized aliases;
- parent tags and optional related tags;
- positive and negative examples;
- lifecycle state: `candidate`, `active`, `merged`, `deprecated`;
- provenance: proposer, source company IDs, discovery run, prompt/model versions;
- created/updated timestamps and ontology version;
- merge/deprecation target where applicable.

Start with useful facets such as customer, industry, workflow, business model, product form, technology, data modality, buyer, go-to-market, deployment, regulation, geography, company stage, and problem archetype.

Do **not** automatically convert metadata fields into semantic tags. Preserve raw and normalized metadata for exact filtering, and independently embed the metadata semantically. Build a deterministic natural-language metadata document per company from publishable field names, values, descriptions, and relationships—for example, batch, year, YC industry/category, location, status, company type, team size, and launch state. Embed that document with the configured embedding model and record the template/version used. Missing metadata must remain missing rather than being inferred by the LLM.

Keep metadata embeddings distinct from LLM-assigned semantic tags. A user must be able to inspect whether a relationship comes from structured metadata similarity, descriptive-text similarity, tag-profile similarity, or the combined company representation.

Implement distinct stages:

1. **Candidate discovery** — Ask the local LLM open-ended questions about small, diverse batches of companies: what reusable semantic attributes distinguish these companies? Require structured JSON and definitions, not merely names.
2. **Normalization** — normalize spelling/casing, detect aliases, and retrieve nearest existing tag definitions using embeddings.
3. **Merge review** — use deterministic thresholds plus LLM adjudication to propose merges; ambiguous merges enter a review queue and never occur silently.
4. **Activation** — activate a candidate only after minimum support, quality rules, and review criteria are met. Allow rare but strategically meaningful tags through an explicit override.
5. **Assignment** — independently evaluate each plausible company/tag pair. Do not evaluate every company against every tag blindly.
6. **Versioning** — snapshot the ontology and preserve migrations so old assignments remain interpretable.

Candidate generation may continue indefinitely, but the active ontology must be curated, deduplicated, measurable, and versioned.

## Tag assignment and vector semantics

Use retrieval to shortlist plausible tags for each company: embedding similarity, facet priors, aliases, metadata rules, and parent/child relationships. Always include a calibrated set of hard negatives.

For each shortlisted company/tag pair, ask the LLM a focused question equivalent to: “Based only on the supplied evidence and tag definition, does this company meaningfully exhibit this attribute?” Require schema-validated JSON containing:

- decision: `yes`, `no`, or `uncertain`;
- confidence from 0 to 1;
- concise rationale;
- evidence source IDs and short evidence spans;
- contradiction or missing-information notes;
- model, prompt, ontology, and run versions.

Never force uncertain evidence into yes/no. Use `yes` assignments to create the primary sparse vector; retain `no` and `uncertain` judgments as audit data.

Define the published feature value explicitly. Default:

`feature_value = calibrated_confidence × tag_information_weight`

where the information weight reduces the dominance of ubiquitous tags. Store the unweighted binary presence and raw confidence separately. Do not mix missing, no, and uncertain values.

Add calibration samples and a human-review set. Measure inter-run stability, duplicate-tag rate, evidence coverage, assignment precision on a reviewed sample, tag prevalence, and orphan-tag rate. Seed all stochastic algorithms.

## Dense embeddings, similarity, and UMAP

Build and version several canonical representations per company, then embed them locally with the configured embedding model:

- `description_embedding`: company descriptions, website-derived summary, and grounded public text;
- `metadata_embedding`: a deterministic text serialization of normalized, publishable metadata;
- `tag_embedding`: canonical names and definitions of positively assigned tags, weighted by assignment confidence;
- `combined_embedding`: a documented, reproducible composition of description, metadata, and tags for the default similarity experience.

Do not merely concatenate vectors from incompatible spaces. Because all representations use the same embedding model and version, combine normalized vectors with configurable weights and renormalize, or embed a canonical combined document; choose one approach, document it, test it, and version its weights/template. Preserve each component vector so the UI can explain and switch similarity modes.

Keep these spaces separate:

- sparse tag vector for interpretability and tag-based similarity;
- dense description, metadata, tag, and combined embeddings for semantic similarity;
- metadata fields for exact filtering;
- 2D UMAP coordinates only for visualization, never as the source of nearest-neighbor truth.

Compute and publish:

- normalized dense vectors or a documented reduced representation if browser payload size requires it;
- top-K nearest neighbors per company for combined, description-only, metadata-only, and tag-embedding spaces, with similarity scores;
- optional top-K neighbors in sparse tag space;
- a reproducible 2D UMAP projection with saved parameters, seed, and run metadata;
- cluster labels generated from overrepresented tags, with honest “algorithmic cluster” wording.

UMAP coordinates will change when the corpus changes. Add an alignment/stability strategy: fit from a stable versioned feature representation, seed the run, and optionally align a new projection to the prior release using shared companies. Never imply that geometric distance in the 2D chart perfectly preserves high-dimensional similarity.

## Incremental pipeline

Implement a Python 3.12 pipeline managed with `uv`. Use typed models and schema validation. A suggested package layout is:

```text
pipeline/
  adapters/
  fetch/
  normalize/
  ontology/
  tagging/
  embeddings/
  projection/
  publish/
  quality/
  cli.py
```

Use content-addressed caches and a manifest/DAG so a stage reruns only when its inputs, configuration, prompt, model, or code version changes. Support checkpointing after every company or bounded batch. Use atomic writes and never corrupt the last successful release.

Provide CLI commands equivalent to:

```bash
uv run yc2vec doctor
uv run yc2vec fetch
uv run yc2vec discover-tags
uv run yc2vec review-tags
uv run yc2vec assign-tags
uv run yc2vec embed
uv run yc2vec project
uv run yc2vec publish-data
uv run yc2vec validate
uv run yc2vec run --incremental
```

Every command needs `--help`, dry-run where meaningful, bounded concurrency, structured logs, progress, retries, resume support, and clear failure messages. Add selectors such as company IDs, batches, changed-since, limit, model, and force-stage.

When a new company appears, process that company and affected indexes only. When a new tag is activated, shortlist candidate companies and evaluate only those pairs plus a statistically useful negative sample. When a tag is merged, apply an explicit migration and recompute only affected sparse vectors and derived artifacts. When the embedding model changes, intentionally create a new embedding-space version and recompute all dependent artifacts.

## Data contracts

Use Parquet for analytical/intermediate tables and JSON/JSONL for inspectable records. CSV is a required export, not the authoritative datastore.

At minimum, maintain versioned schemas for:

- `companies_raw`
- `companies_normalized`
- `web_sources`
- `source_documents`
- `tags`
- `tag_aliases`
- `tag_candidates`
- `source_taxonomy_terms`
- `source_taxonomy_tag_mappings`
- `company_tag_judgments`
- `company_tag_features`
- `company_embeddings`
- `company_neighbors`
- `umap_points`
- `pipeline_runs`
- `release_manifest`

Export:

- `companies.csv`: one row per company with key metadata and summary fields;
- `tags.csv`: ontology records;
- `source_taxonomy.csv`: exact YC/yc-oss categories, industries, source tags, batches, and hierarchy where available;
- `source_taxonomy_tag_mappings.csv`: reviewed relationships between source taxonomy terms and YC2Vec semantic tags;
- `company_tags.csv`: long-form sparse assignments with confidence and provenance;
- optionally `company_tag_matrix.npz` plus row/column maps for efficient research use;
- versioned Parquet files for full-fidelity analysis;
- compact, browser-ready, chunked JSON artifacts.

Do not create a 1,024-column CSV as the only representation. The long-form edge table is the scalable canonical export.

Every public release manifest must include schema version, dataset version, source timestamps, counts, model identifiers, prompt hashes, git commit, data checksums, generation time, known limitations, and licenses/attribution.

## Static publication strategy

GitHub Pages serves only precomputed assets. Keep the published site comfortably below GitHub Pages limits. Do not ship raw website bodies or a full heavyweight analytical database to every browser.

Generate:

- a small boot manifest;
- compact company and tag indexes;
- UMAP point data optimized for progressive loading;
- per-company detail chunks;
- precomputed nearest-neighbor chunks;
- optional batch/year/category partitions;
- Brotli/Gzip-friendly deterministic JSON with short but documented transport keys if needed.

Use hashed assets and a versioned base path. Support both project Pages (`/<repo>/`) and a custom domain. The application must work on direct refresh and under the repository subpath.

## Frontend

Build a polished, responsive, accessible TypeScript application using Vite and React. Prefer a lightweight WebGL/canvas scatterplot library capable of smoothly rendering several thousand points. Avoid rendering one DOM node per company.

Required views and interactions:

### 1. DNA Map

- Large interactive 2D UMAP scatterplot as the primary canvas.
- Pan, zoom, hover, selection, reset, and fit-to-filter.
- Color by batch year, YC category/industry, active semantic tag, or algorithmic cluster.
- Encode selected metadata through color/shape/opacity without making the chart unreadable.
- Search company names instantly.
- Full keyword search across company name, aliases, one-line description, longer summary, canonical tags and aliases, source industries/categories, batch, location, and other publishable metadata.
- Build a compact static search index at publication time and query it entirely in the browser. Use a proven library such as MiniSearch or FlexSearch; do not require a search server.
- Support quoted phrases, multiple terms, typo tolerance where practical, field boosting, and combined keyword + filter queries.
- Rank exact company-name matches first, then tag matches, then descriptions and metadata. Show matched fields and safe highlighted snippets so users understand why a result matched.
- Keep the search index versioned and chunked if necessary; test load time and bundle/data size with the full expected corpus.
- Lasso or rectangle selection if the chosen rendering library supports it robustly.
- A visible explanation that UMAP is a lossy 2D projection.

### 2. Filters

- Batch and year.
- YC source industry/category.
- Semantic tags with AND/OR behavior.
- Status, region, company type, and other available metadata.
- Tag facet and minimum assignment confidence.
- Shareable state in URL query parameters.
- Clear-all and visible result count.
- Keyword search must compose predictably with all filters and map selections, with URL-serializable query state.

### 3. Company detail

- Company name, logo only when redistribution/use is permitted, one-line description, batch, metadata, and source links.
- Assigned semantic tags grouped by facet, confidence, evidence, and rationale.
- Most similar companies ranked by true high-dimensional similarity, not 2D distance.
- Toggle dense-semantic versus tag-profile neighbors.
- Provide explicit similarity modes: Combined, Description, Metadata, Semantic Tags, and Sparse Tag Profile.
- “Why similar” explanation based on shared tags and source-grounded summaries.
- For metadata similarity, show the overlapping or closely related normalized metadata fields; never fabricate a causal explanation from vector proximity alone.
- Coordinates, vector/release version, and data freshness for technical users.

### 4. Tag explorer

- Search and browse canonical tags by facet.
- Definition, aliases, prevalence, lineage, and example companies.
- Show co-occurring tags and distribution across batches/years.
- Deep-link from a tag into the filtered map.
- Show each tag's evidence and assignment provenance. Metadata remains a separate embedded representation and must not be mislabeled as a tag source.

### 5. Search explorer

- Provide a dedicated searchable list/table fallback in addition to map search, so the dataset remains useful without navigating a 2D projection.
- Display ranked companies with match highlights, key metadata, tags, batch, and similarity shortcuts.
- Allow users to move directly from a result to company detail, the filtered DNA map, or similar companies.
- Ensure keyboard navigation, accessible result announcements, debounced input, and useful no-result suggestions.

### 6. About / methodology

- Explain collection, ontology discovery, assignment, embeddings, UMAP, incremental updates, evaluation, limitations, privacy, attribution, and “zero token cost.”
- Clearly distinguish source metadata, model inference, and human-reviewed data.
- Include an unofficial/non-affiliation notice.

### Visual direction

The design should feel like a scientific instrument for startup ecosystems: dark or neutral canvas, crisp typography, restrained accent colors, strong information hierarchy, and no generic AI gradients or excessive cards. Make the map the hero. Provide useful loading, empty, offline, and error states. Ensure keyboard access, focus visibility, reduced-motion support, semantic HTML, sufficient contrast, and mobile fallbacks.

## Repository and automation

Create:

- clear `README.md` with concept, screenshots, local setup, hardware expectations, data flow, CLI usage, GitHub Pages deployment, and limitations;
- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, source attribution, and data-license notes;
- `.env.example` containing no secrets;
- `Makefile` or equivalent task aliases;
- Docker Compose profile for Ollama as an optional convenience, while supporting an existing host Ollama;
- GitHub issue templates for data correction and tag review;
- sample configuration profiles for small, medium, and flagship hardware;
- a small committed fixture dataset so tests and the UI work without downloading models.

Add GitHub Actions:

1. **CI** on pull requests: lint, type-check, unit tests, frontend tests, build, schema validation, and fixture-data smoke test.
2. **Source refresh** on a schedule/manual trigger: fetch public structured records, detect changes, open a data-update PR, and never require a giant local model on a standard hosted runner.
3. **Semantic enrichment** as a manual workflow designed for a labeled self-hosted runner with Ollama. It must checkpoint, upload logs/artifacts, and open or update a PR containing reviewed derived-data changes. Do not expose the runner's Ollama port publicly.
4. **Pages deploy** after validated changes land on the default branch: build the frontend with the correct base path, upload the Pages artifact, and deploy through the official Pages actions.

Do not auto-commit unreviewed ontology merges or low-confidence semantic changes directly to the default branch.

## Testing and quality gates

Implement tests for:

- source parsing and schema drift;
- URL safety and crawler limits;
- prompt-injection resistance in fetched content;
- tag normalization, aliasing, merge migrations, and stable IDs;
- strict JSON repair/retry behavior for Ollama outputs;
- incremental invalidation and resume behavior;
- vector dimensions, normalization, missing-value semantics, and neighbor correctness;
- deterministic UMAP within documented tolerance;
- public artifact schemas and checksums;
- frontend keyword ranking/highlighting, filters, URL state, company detail, search explorer, and tag explorer;
- GitHub Pages subpath routing;
- accessibility smoke tests;
- production build size and successful fixture deployment.

Create a gold evaluation fixture with diverse companies and reviewed tag decisions. CI must not require Ollama: mock its protocol for unit tests and use saved schema-valid fixtures. Add an optional local integration test that calls the configured Ollama models.

Required release gates:

- zero schema-validation failures;
- no missing provenance for published positive tag assignments;
- no NaN/Infinity values;
- every published company has a stable ID and source URL;
- every neighbor refers to an existing company in the same embedding-space version;
- deterministic build output except documented timestamps/manifests;
- frontend works from the configured GitHub Pages project subpath;
- no secrets, raw model responses with sensitive content, or disallowed scraped bodies in the repository.

## Hardware-aware model behavior

At startup, `doctor` must inspect Ollama connectivity, installed models, disk, RAM, and optional GPU information, then estimate whether the selected profile is feasible. Never silently substitute a different model; recommend or require explicit selection.

Profiles:

- `flagship`: an explicitly selected local `qwen3.8-flash-next` 125B variant for tag discovery/adjudication when supported, and `qwen3-embedding:8b` for embeddings. Warn that an MoE model still requires the full quantized weights to be resident across RAM/VRAM and that active-parameter count is not the memory requirement. Validate the exact model license before use or redistribution of outputs.
- `balanced` (default): `qwen3.8:27b` plus `qwen3-embedding:8b` or a smaller Qwen3 embedding model. As of the prompt's authoring, the Ollama `qwen3.8:27b` artifact is approximately 18 GB with a 256K context window; treat this as discoverable runtime information, not a permanent constant.
- `fixture`: no model download; uses committed test responses and data.

All model names, quantization choices, context lengths, timeouts, batch sizes, temperatures, seeds, and endpoints must be configurable and recorded in run manifests. Use low/zero temperature for classification. Validate JSON against schemas and retry only boundedly with repair prompts.

## Implementation sequence

Execute in this order, but do not stop between phases:

1. Inspect repository and environment; document assumptions.
2. Scaffold the typed Python pipeline and React frontend.
3. Define data schemas, manifests, stable IDs, and fixture records.
4. Implement source adapter, normalization, safe optional web enrichment, and cache.
5. Implement Ollama client, ontology discovery/normalization/review, pair shortlisting, and assignment.
6. Implement embeddings, sparse vectors, neighbors, UMAP, a compact static keyword-search index, evaluation, and browser publication.
7. Implement the complete UI against fixture data, then generated data.
8. Add incremental orchestration, CLI, GitHub Actions, documentation, tests, and licensing notes.
9. Run formatting, linting, type checks, unit/integration tests that do not require unavailable hardware, production build, and a local static-server smoke test.
10. Fix all failures, capture a screenshot if browser tooling is available, and report the actual final state.

## Definition of done

The task is complete only when:

- a fresh clone can run the fixture pipeline and UI using documented commands;
- a configured machine with Ollama can run the real incremental pipeline without code changes;
- the fixture dataset produces an interactive map, working filters, company/tag details, and similarity results;
- CSV, Parquet, sparse, and browser-ready outputs are generated and validated;
- the frontend production build succeeds under a GitHub Pages project path;
- CI and Pages workflow files are valid and documented;
- all required tests and quality gates pass locally, except explicitly hardware-gated tests;
- the repository contains no secrets and clearly documents source/data rights and project non-affiliation;
- the final response lists implemented components, commands run, test/build results, remaining limitations, and the exact manual steps required to enable Pages and a self-hosted Ollama runner.

If live data or the flagship model cannot be downloaded in the current environment, complete and validate the entire system using the committed fixture, leave the real adapters operational, and state the exact blocked command. Do not replace real functionality with TODO-only stubs.

---

## Recommended initial defaults

Use these unless repository constraints justify a different choice:

- Python 3.12, `uv`, Typer, Pydantic, Polars/PyArrow, httpx, BeautifulSoup or Trafilatura, NumPy/SciPy, scikit-learn, UMAP-learn.
- React, TypeScript, Vite, a WebGL/canvas scatterplot library, TanStack Query only if it improves static chunk loading, and Vitest/Playwright.
- JSON Schema shared between publication validation and TypeScript-generated types.
- A local manifest-based artifact store before considering DuckDB. DuckDB may be used for local analysis, but it must not become a runtime requirement for the static site.
- Precomputed top-K neighbors and chunked static assets rather than browser-side exhaustive similarity over full 8B embeddings.

When tradeoffs arise, optimize for provenance, incremental recomputation, a compelling map-first experience, and the ability for contributors to add companies or tags safely over time.

