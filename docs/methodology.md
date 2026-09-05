# Methodology

This is the long-form version of the About page: what YC2Vec measures, how, and
what the numbers do and do not support.

## The question

A startup directory tells you which bucket a company was filed under. That is a
fact about the filing, not about the company. YC2Vec tries to describe companies
by properties that cut across those buckets — who buys the thing, what work it
replaces, what data it runs on, how it reaches the customer — and then to make
the resulting structure inspectable rather than asking you to trust it.

## Why an open-ended ontology

A fixed taxonomy is easy to evaluate and always slightly wrong: the interesting
attributes are the ones nobody thought to include. An unbounded one is easy to
generate and useless: it fills with near-duplicates and one-off labels.

The compromise here is *generous proposal, strict activation*. The model may
propose anything, repeatedly, from many different batches of companies. Almost
none of it survives unchanged:

1. proposals are normalised and matched against existing names and aliases;
2. definition embeddings find near-duplicates within a facet;
3. a high similarity merges deterministically, a middle band goes to the model
   for adjudication, and anything still unclear goes to a human queue;
4. a candidate activates only after enough independent companies proposed it and
   its definition is substantive.

The published run started from 3,106 raw proposals, which collapsed to 2,064
distinct candidates, of which 225 merged and **1,018 activated**. The remaining
821 lacked support, and 877 borderline pairs are queued for human review. That
ratio is the point: roughly two-thirds of what the model proposed did not earn
a place in the ontology.

## Why ids are frozen

A tag's id is minted once from its name and never changes. Renaming a tag, or
merging it into another, leaves the id resolvable — a merge sets `merged_into`
rather than deleting the row. Without this, every rename would silently
reinterpret historical assignments, and there would be no way to tell a genuine
change in the data from a change in vocabulary.

## Why shortlisting, and why hard negatives

Judging 6,200 companies against 1,018 tags is 6.3 million model calls, and
nearly all of them would return "no". The shortlist narrows each company to
roughly twenty plausible tags using retrieval similarity, facet coverage,
literal alias hits, metadata rules and tag hierarchy.

That creates an obvious failure mode: if you only ever ask about plausible tags,
a model that says yes to everything looks excellent. So every company is also
judged against a few **hard negatives** — tags sampled from just below the
retrieval cutoff, chosen to look plausible and probably be wrong. The share the
model accepts is a direct estimate of its false-positive rate, and it is
published in `quality.json` rather than kept internal.

## Why `uncertain` is a real answer

Forcing a binary decision on thin evidence manufactures confidence. The schema
allows `yes`, `no` and `uncertain`, and the prompt explicitly invites the third.
Two further checks run after the model answers:

- every quoted evidence span must literally occur in the document it cites;
  paraphrased or invented quotes are discarded;
- a positive left without any surviving evidence is downgraded to `uncertain`.

A release gate then refuses to publish a positive assignment that has no
evidence and no rationale, so the published sparse vector contains only claims
that can be traced to a sentence.

## The published feature value

```
feature_value = calibrated_confidence × information_weight
```

`calibrated_confidence` applies a mild concave correction to the model's raw
score, which clusters near 1.0. `information_weight` is `log(1/prevalence)`
normalised into `(floor, 1]`, so a tag that applies to almost every company
contributes almost nothing while a rare one approaches full weight.

Binary presence, raw confidence, calibrated confidence and the weight are all
stored separately. Nothing is collapsed, and missing, `no` and `uncertain` are
never conflated with each other.

During a partial run the prevalence denominator is the number of companies
actually judged, not the corpus size — otherwise every tag would look rare and
every weight would be inflated.

## The vector spaces

Four dense representations, all from the same embedding model and version:
description, metadata, tag profile, and a weighted combination. Because they
share a space, combining them is a weighted sum of normalised vectors followed
by renormalisation, not a concatenation of incompatible spaces. Missing
components (a company with no tags) are dropped and the remaining weights
renormalised, so a sparse company is not dragged toward the origin.

A fifth space is deliberately *not* dense: weighted Jaccard over the
interpretable sparse tag vector. It answers "which companies were given the same
tags?", which is a different question from "which companies embed similarly",
and users can switch between them explicitly.

## What the map is and is not

UMAP is a lossy projection into two dimensions. Local neighbourhoods are
broadly meaningful; global distances are not. Nothing in the product reads
similarity from the map — every "similar companies" list comes from precomputed
exact top-K in the full space, and the projection stage never writes
neighbours.

Coordinates also move when the corpus changes, because UMAP's output is only
defined up to rotation and reflection. Each run is seeded, its parameters are
hashed into a `projection_version`, and a new projection is aligned to the
previous release with an orthogonal Procrustes fit over the companies present in
both. That removes the arbitrary rotation without distorting relative positions.

Clusters are KMeans over the 2D coordinates, labelled by tag **lift** — the
in-cluster rate divided by the global rate — rather than raw frequency, which
would label every cluster with the most common tag in the corpus. They are
algorithmic groupings and the UI says so.

## Source taxonomy versus inferred ontology

YC's own industries, tags and batches are ingested with their exact names and
ids and are never renamed, merged or absorbed into the ontology. A separate
mapping table records how each inferred tag relates to a source term
(`equivalent`, `broader`, `narrower`, `overlaps`, `related`) so the two can be
compared. Metadata is also embedded on its own, as a deterministic natural
language document, and is exposed as exact filters — it is never converted into
a semantic tag.

## What would change the numbers

- **A different chat model** changes the ontology and the assignments, and
  should be treated as a new dataset rather than an update.
- **A different embedding model** creates a new embedding-space version and
  invalidates every vector, neighbour list and projection, while leaving the tag
  judgments intact.
- **A prompt edit** invalidates exactly the stage that uses that prompt, because
  prompt text is hashed into the cache key.
- **More companies judged** changes tag prevalence and therefore every
  information weight, which is why the release manifest records counts.

## Honest limitations

- Tag assignments are model inference. The evidence spans let you check any one
  of them; they do not make the set as a whole correct.
- Coverage is uneven while assignment is incremental. A company with no tags has
  not been judged yet, or had too little public text to ground anything.
- The gold review set is small and hand-written. It catches regressions on
  obvious cases; it does not estimate accuracy on hard ones.
- Generation is not bit-reproducible across hardware and runtime versions even
  at zero temperature with fixed seeds. The deterministic parts —
  normalisation, vector arithmetic, seeded UMAP, publication — are.
