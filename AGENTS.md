# AGENTS.md

Project memory for AI coding agents working in this repository.

## Project overview

A Slack bot that acts as a research helper for real-estate finance teams. It
takes an ambiguous question, scopes it with clarifying follow-ups, prioritises
the relevant data sources, retrieves across them, and returns a cited answer.
It protects PII, exposes the SQL it generates on demand, and escalates to a
human reviewer when confidence is low or the question needs judgement.

## Tech stack

- **Language:** Python 3.12
- **Slack:** `slack_bolt` (Socket Mode for the demo; HTTP events for Cloud Run)
- **LLM:** Anthropic Claude (tiered: Haiku for routing/scoping, Sonnet for SQL + synthesis)
- **Embeddings:** local `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim)
- **Datastore:** Postgres + `pgvector` (Docker)
- **Config:** `python-dotenv`
- **Tooling:** `ruff`, `pytest`, GitHub Actions

## Key conventions

- All LLM calls go through `llm/client.py`; never call a provider SDK directly
  from feature code. This is what makes the token/cost logger reliable.
- Every retrieval source implements the `DataSource` protocol in `sources/base.py`.
- Secrets live in `.env` (gitignored). Never log secrets or PII.
- Conventional commits, one logical change per commit, feature branches + PRs.
- Prompt or agent-pipeline changes require an eval run
  (`python scripts/eval_questions.py`) before merging; paste the pass-rate
  report into the PR body. Answer quality is measured, not eyeballed.
- Raw datasets are gitignored; only ingestion scripts and redacted samples are committed.

## Environment and commands

```bash
# Create/activate venv (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run tests
pytest

# Lint / format
ruff check .
ruff format --check .

# Run the bot (once implemented)
python -m research_agent.app
```

## Data pipeline

```bash
python -m research_agent.ingest          # Airbnb + Land Registry -> warehouse
python -m research_agent.transactions    # modelled finance ledger
python -m research_agent.policy_kb       # gov.uk policy knowledge base
python -m research_agent.playbooks       # source registry + embeddings
python scripts/load_review_embeddings.py # load Colab GPU review vectors
```

Raw data lives in gitignored `data/raw/`; only `data/seed/` (PII-redacted) is committed.

## Do not touch

- `data/raw/` — downloaded real data, never committed or edited.
