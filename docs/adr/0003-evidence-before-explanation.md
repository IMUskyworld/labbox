# ADR 0003 — evidence before explanation

Date: 2026-09-22
Status: accepted
Relates to: DEVELOPMENT.md §1.2, §10.5, §11, `llm/context.py`, `llm/provider.py`

## Context

An LLM asked "why did accuracy drop?" will always produce a confident sentence. Training
logs, Git commit messages and file paths are attacker-controlled input, and a model that
reads them can be instructed to do anything a tool-calling agent can. Meanwhile a wrong
causal claim in a lab notebook costs more than no claim.

## Decision

The model is a labeller, not an investigator:

1. **Facts first.** Rules, collectors and the diff engine decide what happened. The model
   receives only an `ExplainContext`: structured facts with ids, diagnostics, selected
   metric values, and the list of fields that were never captured.
2. **Nothing else leaves the machine.** No raw logs (unless `send_logs = true`), no
   environment values, no file contents, no config files. `--dry-run` prints exactly which
   categories would be sent, before any request.
3. **Citations are enforced after the fact.** Every `fact_id` the model cites is checked
   against the ids that were actually issued; unknown citations are stripped and recorded
   in `rejected_fact_ids`. A hypothesis citing an invented id is dropped whole — trimming
   the bad id would launder an unsupported claim into the report.
4. **No causal vocabulary.** Rules phrase multi-variable changes as "cannot attribute", and
   a test asserts the word "caused" never appears in a comparison payload.
5. **Output must validate.** Response is parsed against `ExplainResult`; on failure the
   model output is discarded and only the deterministic analysis is shown.
6. **Kill switch.** With `enabled = false` the whole layer is a `NullProvider` and every
   other command behaves identically — verified by the M4 acceptance tests.

## Consequences

* The AI feature is deliberately weaker than a generic agent: it cannot look anything up,
  so a vague answer is possible. That is the trade we chose.
* Report pages label the model section as model-generated, visually separate from evidence.
* Any future "let the model read a file" request needs a new ADR, not a code change.
