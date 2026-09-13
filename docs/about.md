# About the Research Agent

A research helper for real-estate finance teams. Ask it a question in Slack; it
clarifies with you if needed, picks the right data sources, queries them, and
answers with citations. It shows the SQL it ran, protects personal data, and
escalates to a human reviewer only when judgement — not clarification — is
required.

## Data sources

| Source | What's in it | How it's queried |
|---|---|---|
| **Property & leasing warehouse** | 92,638 London short-let listings, nightly prices, availability (33.8M calendar rows) | SQL |
| **Finance ledger** | 5.5M modelled invoices, payments, fees, VAT, refunds, chargebacks — with overdue/partial/pending states | SQL |
| **HM Land Registry** | 64,213 real London property sales (2025), prices, postcodes, tenure | SQL |
| **Guest reviews** | 2.24M real review texts, semantically searchable | Vector search |
| **Policy KB** | Official gov.uk guidance: deposits, evictions, landlord obligations | Vector search |

Sources are linked: reviews, calendar and the ledger share `listing_id`; Land
Registry joins at borough level. Answers show a _Sources_ footer naming what
was used, and the agent can combine sources — e.g. SQL finds the top-price
listings, then review search runs scoped to exactly those properties.

## How to interact

- **@Research Agent <question>** in any channel it's in — it replies in-thread.
- **Reply in that thread** (no @mention needed) to refine or follow up — it
  remembers the conversation and re-runs with your direction.
- **Ambiguous questions get clarifying questions back** — answer in-thread and
  it proceeds. It asks you, not a reviewer: intent is yours.
- **Thin evidence gets a best-effort answer** that names what's missing, so you
  decide whether to refine or involve a reviewer.
- **Judgement questions** (policy, legal, commercial calls) escalate to the
  review channel. Reply with guidance there (or in the original thread) and the
  agent **acts on it**: it re-runs the analysis and posts the outcome back to
  you, attributed _under reviewer direction_ — up to four rounds of review.

## Commands

- `/sql` — shows the last SQL query the agent ran in this channel, so any
  number can be independently verified.
- `/progress` — on demand: what the agent is working on right now (with
  elapsed vs typical timing), data loaded, recent questions here, pending
  reviews, and model calls/cost to date.

## Privacy & safety

- Personal data in review text is redacted before it reaches the model or logs.
- Generated SQL is read-only, single-statement, table-restricted, row-limited
  and time-capped; it can never modify data.
- Every model call is logged with token counts and estimated cost.

## Suggested conversations

**Portfolio finance (SQL)**
- "What is our overdue rent across the portfolio?"
- "How much did we collect in host payouts last month, and what's still pending?"
- "Which neighbourhoods have the highest overdue balances?"

**Market comparison (SQL, cross-source)**
- "Compare our average nightly price in Camden against actual sold prices there."

**Guest experience (semantic)**
- "What have guests complained about regarding cleanliness?"
- "What do guests say about check-in difficulties?"

**Multi-source (SQL scopes the semantic search)**
- "What are the most common issues reported by guests in the top 10% most
  expensive airbnb locations?"

**Policy (semantic, cited)**
- "How long do we have to protect a tenancy deposit?"
- "What notice is required to evict a tenant?"

**Scoping (clarifying questions)**
- "How are we doing?" — it will ask what you mean before answering.

**Human-in-the-loop (reviewer-driven)**
- "Should we evict a tenant who is three months behind on rent?" — judgement
  call: watch it escalate, then steer it with reviewer guidance.
