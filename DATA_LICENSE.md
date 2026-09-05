# Data sources, licences and redistribution review

YC2Vec is an unofficial project. It is not affiliated with, endorsed by, or
connected to Y Combinator. This document records what we redistribute, on what
basis, and what we deliberately do not.

## Review checklist

Every source must clear all five before it may be enabled:

1. **Provenance** — the source is identified, pinned, and its retrieval time recorded.
2. **Terms** — the licence or terms of use permit the specific use we make of it.
3. **Redistribution** — if rights are unclear, we store only derived facts,
   hashes, citations and URLs, never the original body.
4. **Access controls** — nothing bypasses authentication, paywalls or rate limits.
5. **Opt-out** — there is a documented way for a company to be excluded.

## Source 1: `yc-oss/api` (enabled)

- **What it is:** an open-source project that republishes Y Combinator's public
  company index as static JSON.
- **URL:** <https://github.com/yc-oss/api>, served from <https://yc-oss.github.io/api>.
- **Licence:** MIT (the project's own code and published data files).
- **What we take:** the structured company records and the classification
  collections (industries, tags, batches).
- **What we redistribute:** normalised metadata and company descriptions, with
  attribution and a link back to the YC company page for every record.
- **Review:** the records describe companies, not individuals, and are already
  published publicly by Y Combinator. We reproduce YC's own classification names
  and ids exactly and never rename or merge them.

**Y Combinator's own rights are not transferred by this project.** Company
names, logos and descriptions remain the property of the companies described.
Logos are hotlinked from the source record rather than copied into this
repository, and the site renders the company name as text so a blocked image
costs nothing.

## Source 2: `company_website` (disabled by default)

- **What it is:** the company's own public homepage and up to two same-origin pages.
- **Status:** opt-in. It is off unless `YC2VEC_ENABLE_CRAWL=true`.
- **Review:** page bodies are third-party copyrighted content with unknown
  redistribution rights, so **raw page bodies are never committed**. Only
  extracted main text is kept, and only in the local cache under `data/cache/`,
  which is git-ignored. Published artifacts contain at most short quoted
  evidence spans, which is the same fair-dealing basis as a search snippet.
- **Conduct:** the crawler obeys `robots.txt`, identifies itself with a
  transparent user agent and a project URL, throttles per domain, caps response
  size and content type, restricts and re-validates redirects, refuses private
  and cloud-metadata addresses, and never executes page JavaScript.

## Derived data

The ontology, tag assignments, embeddings, neighbours and projection are
produced by this project and released under **CC BY 4.0**. They are model
inference, not fact, and are labelled as such throughout the site.

## Opt-out

To have a company excluded from the published dataset, open an issue using the
data-correction template, or add its domain to `crawl.denylist_domains` for
crawl-only exclusion. Removal requests are honoured without argument.

## Attribution

> Company records derive from the open-source yc-oss/api project, which
> republishes Y Combinator's public company index. YC2Vec is an independent,
> unofficial project and is not affiliated with or endorsed by Y Combinator.
