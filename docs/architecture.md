# Architecture

## The shape of the system

```
  public source                offline pipeline                    static site
 ┌──────────────┐   ┌────────────────────────────────────┐   ┌──────────────────┐
 │ yc-oss/api   │──▶│ fetch ──▶ normalize                │   │  GitHub Pages    │
 │ (JSON)       │   │            │                       │   │                  │
 └──────────────┘   │            ├─▶ discover-tags       │   │  points.json     │
 ┌──────────────┐   │            │      │                │   │  companies.json  │
 │ company site │┄┄▶│            │   review-tags         │──▶│  tags.json       │
 │ (opt-in)     │   │            │      │                │   │  detail/*.json   │
 └──────────────┘   │            │   assign-tags         │   │  search/docs.json│
                    │            │      │                │   │                  │
                    │            └─▶ embed ─▶ project    │   │  no backend      │
                    │                   │                │   │  no database     │
                    │            publish-data ─▶ validate│   │  no model        │
                    └────────────────────────────────────┘   └──────────────────┘
                              Ollama, local only
```

Everything expensive happens on the left. The browser receives precomputed
files and does filtering, search and rendering locally.

## Why the stages are separated this way

**Fetch and normalise are one stage, discovery is another.** Normalisation is
deterministic and cheap; discovery costs model time. Keeping them apart means a
source refresh does not invalidate the ontology.

**Discovery, merge review and activation are three stages, not one.** An
open-ended ontology needs generosity at proposal time and discipline at
activation time. Collapsing them produces either a thin ontology or a swamp of
near-duplicates. The merge stage is where duplicates die, and it is deliberately
three-tiered: deterministic auto-merge above a high similarity, model
adjudication in a middle band, and a human review queue for anything still
ambiguous. Merges are never silent.

**Shortlisting is separate from judging.** Judging every company against every
tag is quadratic and mostly wasted — the overwhelming majority of pairs are
obvious negatives. Retrieval, facet priors, alias hits, metadata rules and tag
hierarchy narrow the field; hard negatives keep the result measurable.

**Embedding is separate from assignment** even though both need a model, because
they need *different* models and invalidate on different inputs. Changing the
embedding model creates a new embedding-space version and recomputes vectors,
neighbours and the projection — but leaves every tag judgment intact.

## Incrementality

The store hashes each stage's declared inputs — upstream artifact hashes, the
relevant config slice, prompt text, model identity and the pipeline version —
into a *stage key*. A stage whose key matches its recorded key is skipped.

That means:

- a new company reruns normalisation for that company and the indexes it touches;
- a prompt edit invalidates exactly the stage that uses that prompt;
- a model change invalidates everything downstream of it;
- a code change that alters semantics invalidates on the version constant you bump.

Long stages also checkpoint after every company into a `.partial.jsonl`, so an
interrupt costs one company rather than the run. Writes are atomic: the previous
release survives a crash mid-publish.

## The four vector spaces, and why they stay separate

| Space | Built from | Answers |
| --- | --- | --- |
| `description` | the company's own prose | "who writes about themselves this way?" |
| `metadata` | a deterministic serialisation of structured fields | "who shares this shape?" |
| `tags` | definitions of assigned tags, ordered by feature value | "who has this semantic profile?" |
| `combined` | weighted sum of the three, renormalised | the default similarity experience |

All four come from the same embedding model and version, which is what makes
the weighted sum legitimate — combining vectors from different models would be
meaningless arithmetic. Component vectors are all preserved so the UI can switch
modes and say which one it is using.

A fifth space, the **sparse tag profile**, uses weighted Jaccard over the
interpretable feature vector rather than a dense embedding. It answers a
genuinely different question: "which companies were given the same tags?"

The 2D UMAP projection is in none of these lists on purpose. It is for looking
at, and nothing reads similarity from it.

## What the browser gets

| File | Purpose | Loaded |
| --- | --- | --- |
| `manifest.json` | versions, counts, checksums, shard layout | first |
| `points.json` | parallel typed arrays for the scatterplot | first |
| `companies.json` | compact index for filters and lists | first |
| `tags.json` | ontology with prevalence and co-occurrence | first |
| `taxonomy.json` | exact YC terms and their mapping to our tags | first |
| `clusters.json` | algorithmic cluster labels | first |
| `search/docs.json` | projected search documents | after paint |
| `detail/<0-63>.json` | full company records with evidence and neighbours | on demand |

Company detail is sharded so opening one company fetches roughly 1/64th of the
detail data rather than all of it. The shard function is duplicated in
`pipeline/publish/browser.py` and `frontend/src/lib/data.ts`, and a test pins
them to each other.

## Things that would be easy to get wrong

- **Reading similarity off the map.** The UI never does; the disclaimer says so.
- **Turning metadata into tags.** Metadata is filtered exactly and embedded
  separately. It is never converted into a semantic tag.
- **Letting a raw model string become an identifier.** Names are display-only;
  ids are minted once and frozen.
- **Mixing embedding-space versions.** A release gate fails if neighbours span
  more than one.
- **Conflating missing, `no` and `uncertain`.** They are three different states
  and are stored as three different states.
