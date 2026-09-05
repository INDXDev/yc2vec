# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/INDXDev/yc2vec/security/advisories/new).
Please do not open a public issue for a vulnerability.

Expect an acknowledgement within a week. There is no bounty programme.

## What is in scope

YC2Vec is a static site over a precomputed dataset, plus an offline pipeline.
The interesting attack surface is small but real:

**The website crawler (`pipeline/adapters/company_website.py`).** It follows
URLs supplied by an upstream dataset, so it is an SSRF surface. Every URL and
every redirect target passes through `check_url`, which rejects non-HTTP(S)
schemes, non-web ports, private, loopback, link-local, reserved and
cloud-metadata addresses, and resolves hostnames so that a public name pointing
at a private address is caught. A bypass of any of these is a vulnerability.

**Prompt injection (`pipeline/prompts.py`).** Company descriptions and website
text are untrusted third-party content. They are fenced in `<untrusted_*>`
delimiters that the content cannot close, and every system prompt states that
content inside those delimiters is data rather than instructions. A payload
that escapes the fence, changes the output schema, or causes the pipeline to
take an action is a vulnerability. Note that a payload which merely persuades
the model to assign a wrong *tag* is a data-quality bug, not a security one —
report it as a data correction.

**Published artifacts.** The site renders dataset text as text, never as HTML,
and no dataset field is interpolated into markup. An injection through
published data would be a vulnerability.

## What is out of scope

- The Ollama server. It has no authentication by design and must never be
  exposed publicly. Binding it to a public interface is a deployment mistake,
  not a YC2Vec vulnerability.
- Inaccurate semantic tags. These are model inference; use the data-correction
  issue template.
- Denial of service against a self-hosted runner.

## Secrets

The pipeline requires no credentials: every model call runs locally. There are
no API keys in this repository, and `.env.example` contains none. If you find a
committed secret, report it as a vulnerability.
