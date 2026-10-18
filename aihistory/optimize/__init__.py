"""`aihistory optimize`: test, and optionally tune, the text that steers your coding agents.

The first target is skill trigger descriptions: does the agent load the right skill for a
prompt you actually wrote? The steps:

1. `mine`: candidate prompts from your own history, with the skill each one led to;
2. `label`: you check the pre-filled labels in LABEL.md (each says where it came from: your
   history, a generated prompt, or a judge) and tick `I checked this`; only checked labels go
   into the test split;
3. `eval`: a headless agent runs every prompt against each arm of descriptions;
4. `report`: accuracy per arm against a hand-written control, with McNemar tests.

Every agent run is written as an agent-trace session (`traces/`) and scored from its events,
so Claude Code and Codex runs are judged the same way, and any run can be replayed.
"""
