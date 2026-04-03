---
name: run_all_script_fix_failures
description: Use when you need to execute Python files under scripts/ in this PyAutoGalaxy workspace, follow PyAutoBuild's autogalaxy no_run policy, run in test mode by default, and fix failing scripts in place as you go without maintaining script_errors.md, script_errors_full.md, or .script_runner state.
---

# Run All Scripts And Fix Failures

Use this skill in the autogalaxy workspace when the user wants an iterative validation pass across `scripts/**/*.py` where each failing script is fixed immediately and rerun before moving on. This skill is non-interactive: continue working without pausing for user input unless a real blocker prevents safe progress.

Match PyAutoBuild behavior instead of inventing a separate policy:

- Respect `../PyAutoBuild/autobuild/config/no_run.yaml` for project `autogalaxy`.
- Use PyAutoBuild ordering: if a folder contains `start_here.py`, run it before the other scripts in that folder and before descending into subfolders.
- Otherwise keep the run order stable and deterministic.

## Workflow

1. Read `CLAUDE.md` in the repo root if it has not been read this turn.
2. Run from the repository root so relative dataset and output paths resolve correctly.
3. Use test mode by default:

```bash
PYAUTOFIT_TEST_MODE=1 python <script.py>
```

4. Build the script list using the PyAutoBuild skip policy and the workspace ordering above. Do not use `run_all_scripts.sh`, because that workflow is designed around continuing after failures and writing failure logs under `failed/`.
5. Run one script at a time.
6. If a script fails, inspect the traceback, patch the script or the minimum necessary local code, and rerun that same script before continuing.
7. If a fix is not feasible in the current turn, stop and report the blocker instead of creating a long failure log.
8. Surface progress in the Codex window as you go rather than behaving like a silent batch job.

## Behavior

- Do not write logs under `failed/` as part of this skill.
- Keep fixes narrowly scoped to the failing behavior you reproduced.
- Prefer updating files under `scripts/`; notebooks under `notebooks/` are generated artifacts and should not be edited.
- When a failure appears to come from upstream library code or missing data rather than the example script, verify that before making broader changes.
- After each fix, rerun the affected script with the same environment to confirm the failure is resolved.
- For each attempted script, post concise commentary updates in the Codex window:
  - Announce which script is running.
  - If it fails, show the key error or final traceback line.
  - Before editing, state the planned fix in one or two sentences.
  - After editing, report the verification rerun result.
  - If a fix is not feasible, say why you are stopping.
- Do not ask the user to confirm each fix or next step. Keep moving and only stop when you hit a genuine blocker or a risky ambiguous change.

## Notes

- Test mode reduces search work but does not guarantee every script will finish quickly.
- Some scripts need `if __name__ == "__main__":` guards for JAX plus multiprocessing. See `scripts/guides/modeling/bug_fix.py` when a fit crashes during parallel startup.
- This skill is for active repair, not bulk reporting. Keep the terminal output focused on the current failing script, the chosen fix, and the verification rerun.
