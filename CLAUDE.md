## Model & token use

- Default model for this project: **Opus (`opus`)**, set in `.claude/settings.json`; mesh-cutting geometry is genuinely hard. Helper agents (subagents) run on Haiku.
- At the start of a session, and whenever the task changes, check whether the model you are running fits the job. If it doesn't, tell Peter in one line which model to switch to and that he can type `/model <name>` (you cannot switch it yourself). Guide:
  - `haiku`: file searches, bulk data/CSV work, simple renames.
  - `sonnet`: routine edits, docs, copy, small UI tweaks, prompt writing.
  - `opus`: debugging, architecture, hardware/printer/serial/EFTPOS, firmware, security, anything that has failed once already.
  - `fable`: never by default. Peter is on the Pro plan, where Fable is billed as extra usage credits, not plan limits. Suggest it only after Opus has failed twice on the same problem, and say it costs extra.
- Keep context small: open the specific files a task needs rather than sweeping the repo.
