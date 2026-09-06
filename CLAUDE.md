# statgpu Claude Code bootstrap

This file is a small routing shim, not a second copy of the project rules.

For repository implementation, review, benchmark, validation, or user-facing capability work:

1. Read `dev/AGENTS.md` before making project-contract decisions.
2. Use the canonical task procedure under `.claude/skills/<skill-name>/SKILL.md` when one applies.
3. Treat `.claude/workflows/` as the namespace for Claude Code Dynamic Workflow scripts, not as a documentation/archive directory.
4. Treat `dev/reviews/` and superseded plan text as historical evidence unless a current plan explicitly says otherwise.

Canonical task skills:

- `.claude/skills/new-module-dev/SKILL.md`
- `.claude/skills/code-review/SKILL.md`
- `.claude/skills/benchmark/SKILL.md`

`dev/AGENTS.md` is the project guide; the skills are procedure-specific. Do not duplicate their detailed gates here.
