# Data sources, schema, and ethics

## DEMO data

`src/demo_data.py` generates synthetic URLs using only reserved names:

- `example.com`, `example.net`, `example.org` (RFC 2606)
- `example.test` subdomains (RFC 2606)
- `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24` (RFC 5737 documentation ranges)

Labels come from a lexical rule plus 8% noise. **No URL points to a real
phishing site and no claim about real detection is possible from this data.**
Three synthetic "sources" deliberately differ:

| Source | Benign host | Phishing pattern |
|---|---|---|
| `demo-feed-a` | `example.com` | hyphenated subdomains, `/login/<id>` |
| `demo-feed-b` | `example.net` | IP host + non-standard port, `.php` |
| `demo-feed-c` | `example.org` | long `.example.test` subdomains, numeric paths |

Generate:

```bash
python scripts/make_demo_data.py --n-urls 6000
```

## Real data (optional)

`load_dataset(source="csv", raw_dir="path/to/phishing_urls.csv")` expects:

| Column | Type | Meaning |
|---|---|---|
| `url` | string | Full URL (`http`/`https`) |
| `label` | 0/1 | 1 = phishing/malicious, 0 = benign |
| `source` | string | Feed or dataset name (required for cross-source split) |
| `timestamp` | ISO-8601 | First-seen or collection time |
| `url_id` | string, optional | Stable identifier; generated if missing |

Verify a local file:

```bash
python scripts/download_phishing_data.py --path data/raw/phishing_urls.csv
```

Possible sources (check licence, terms, and label quality):

- **PhishTank** — community-verified phishing URLs; requires an account/API key
  and attention to verification status and takedown time.
- **OpenPhish** — public feed; links are live malicious URLs. Never visit them;
  handle as untrusted text.
- **URLhaus (abuse.ch)** — malware distribution URLs; terms apply.
- **Tranco / curated lists** — for benign negatives. Domain rank lists are not
  automatically benign; sampling and verification matter.

Do **not** commit downloaded datasets. Do not redistribute feeds whose terms
forbid it. Do not execute or fetch suspicious URLs in the notebook/script.

## Label and temporal caveats

- Phishing URLs are short-lived. A URL that is offline today may have been
  active when collected.
- Feed labels mix automated detection and human verification; false positives
  and delayed negatives exist.
- `timestamp` semantics differ by source (first seen, submission, verification).
  This makes temporal splits approximate unless the collection protocol is
  harmonised.
- Cross-source tests confound domain shift, labelling policy, and collection
  period. They are a stress test, not a causal decomposition.
- Benign negatives must come from a comparable population; using only homepage
  URLs inflates lexical separability.

## Ethics and safety

- Never open suspicious URLs outside an isolated environment.
- Do not attempt credential entry or interaction with phishing pages.
- Avoid storing personal data embedded in URLs (emails, tokens, query params).
- This repository is a research prototype, not a phishing blocklist or an
  operational security control.
