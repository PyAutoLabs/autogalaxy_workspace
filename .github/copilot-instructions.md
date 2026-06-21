# Copilot Coding Agent Instructions

The canonical, agent-agnostic instructions for this workspace live in [`/AGENTS.md`](../AGENTS.md).
Read that file first — it covers repository structure, running and testing scripts, notebook
generation, bulk-edit safety, and the API-update / general-issue / PR-description task workflows.

Copilot-specific reminders:

- Only edit files in `scripts/`. Never edit files in `notebooks/` — those are auto-generated.
- Do not add new scripts unless the issue specifically asks for it; for API updates, only update
  existing scripts to match the changed API.
- Preserve all docstrings, comments, and tutorial explanations; change only code that uses the old
  API.
