# real-estate-research-agent

A Slack research agent for real-estate finance teams. It scopes an ambiguous
user question, prioritises the most relevant connected data sources, retrieves
across structured and unstructured data, answers with citations, reveals the SQL
it ran, protects PII, and escalates to a human when confidence is low or human
judgement is required.

Built as a forward-deployed-engineer case study: understand the problem, scope
the solution, build, deploy, iterate.

## Status

Early scaffold. See `AGENTS.md` for structure and conventions.

## Data

Runs on real, openly-licensed data:

- **Inside Airbnb (London)** — listings, calendar and reviews, keyed by
  `listing_id`. Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **HM Land Registry Price Paid Data** — contains HM Land Registry data
  © Crown copyright and database right, licensed under the
  [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- **UK public guidance** (OGL v3.0) — used as the policy knowledge base.

Raw data is downloaded at setup into a gitignored `data/raw/` directory and is
never committed. A small, PII-redacted sample is committed for tests.

A statistically modelled transaction ledger (regression + noise) is generated
from the real occupancy and pricing data to provide realistic finance
operations (invoices, payments, refunds, arrears).

## Pipeline

```bash
# 1. Warehouse: full London listings + 12 months of Price Paid, plus entity resolution
python -m research_agent.ingest

# 2. Modelled finance ledger (fees, VAT, payments, refunds, chargebacks)
python -m research_agent.transactions

# 3. Policy knowledge base from gov.uk guidance (OGL v3.0)
python -m research_agent.policy_kb

# 4. Seed the source registry (playbooks) + embeddings
python -m research_agent.playbooks

# 5. Review embeddings: bulk on a Colab GPU (same bge-small model as queries)
#    Run scripts/colab_embed_reviews.py in Colab, download shards, then:
python scripts/load_review_embeddings.py
python scripts/check_embedding_compatibility.py   # GPU vs local vectors agree
```

Review vectors are produced on a GPU with `BAAI/bge-small-en-v1.5` and queried
locally with `fastembed` using the **same model**, so no vector schema change or
runtime API key is required. Query-time retrieval stays local; only the bulk
embedding job runs in Colab.

