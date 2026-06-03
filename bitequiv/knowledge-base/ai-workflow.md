# AI Coding Workflow & Knowledge Persistence

How to keep `CLAUDE.md` and other AI-coding artifacts useful over the 12-week
project, and how to persist them with git so nothing is lost between sessions.

## 1. Directory layout

```
bitequiv/
├── CLAUDE.md                         # project-scoped guide (loaded with repo-root CLAUDE.md)
├── knowledge-base/
│   ├── ai-workflow.md                # this file
│   ├── tree-reduction-in-ptx-and-triton.md   # teaching/reference docs (one concept each)
│   └── claude-chatting/
│       └── YYYY-MM-DD-<topic>.md     # dated session summaries
```

- **`CLAUDE.md`** = stable, high-signal context that should be in *every*
  session's working memory. Keep it short; link out to detail.
- **`knowledge-base/*.md`** = deep reference docs. One concept per file. Not
  auto-loaded — pulled in on demand or linked from CLAUDE.md.
- **`claude-chatting/*.md`** = append-only log of what happened each day.

## 2. The two-tier rule

Think of it as a cache hierarchy:

| Tier | Lives in | Loaded when | Holds |
|------|----------|-------------|-------|
| Hot | `CLAUDE.md` | every session, automatically | goals, milestones, conventions, guardrails, "what exists vs. build", open questions |
| Cold | `knowledge-base/` | on demand / via links | full explanations, PTX walkthroughs, design notes, session history |

Promote a fact to `CLAUDE.md` only when *most* future sessions need it. Demote
detail to `knowledge-base/` to keep `CLAUDE.md` lean. If `CLAUDE.md` grows past
~2 screens, it's doing too much — move detail out.

## 3. When to update what

- **End of a working session** → ask "summary today's work" / "summary today's
  chat". Produces `claude-chatting/YYYY-MM-DD-<topic>.md` (template below).
- **Learned a reusable concept** (a PTX pattern, a compiler-pass behavior, a
  numerics gotcha) → new or updated `knowledge-base/<concept>.md`.
- **Changed a goal, convention, or discovered key infra** → edit `CLAUDE.md`
  (Sections 3, 5, 6, 9 especially).
- **Finished/triaged a known bug or design decision** → update the relevant
  knowledge-base doc *and* the "Open questions" list in `CLAUDE.md`.

Before creating a new file, check for an existing one that covers it and update
that instead — avoid duplicates.

## 4. Session-summary template

`knowledge-base/claude-chatting/YYYY-MM-DD-<short-topic>.md`:

```markdown
# YYYY-MM-DD — <Title>

## What we covered
### 1. <Theme>
- bullet points of what was discussed/done/decided

## Key diffs and file paths referenced
- D-numbers (e.g. D100027220 — description)
- absolute file paths touched or studied

## Documents created
- new/updated knowledge-base files

## Open questions for follow-up
- things to resolve next session
```

Rules: use the **actual date** (today is provided in session context — don't
guess), convert relative dates ("yesterday") to absolute, and record the things
a future session *cannot re-derive* — D-numbers, paths, decisions, dead ends.

## 5. Git persistence

Everything under `bitequiv/` is committed so it survives across sessions,
machines, and context resets.

- **Commit cadence:** commit knowledge-base + CLAUDE.md updates at the end of
  each session (or alongside the code change they document). Small, frequent
  commits beat one big dump.
- **Suggested message style:** `bitequiv(kb): <what changed>` for knowledge,
  `bitequiv(claude): <what changed>` for CLAUDE.md, normal style for code.
- **Don't auto-commit:** per repo policy (`CLAUDE.md` → "Don't commit unless the
  user explicitly asks"). The agent prepares the change; you trigger the commit.
- **Branching:** `bitequiv/` notes can ride along on whatever feature branch
  you're on, or live on a long-lived notes branch — but keep them in the repo,
  not in scratch dirs, so they're never lost.
- **Disclose AI authorship** in PR descriptions when code was Claude-authored.

## 6. Practices that keep AI sessions productive

- **State constraints up front.** "This must stay bitwise-equivalent under
  ordered reduction" belongs in the prompt *and* in code comments — see CLAUDE.md
  §6. The agent will otherwise optimize the constraint away.
- **Correctness before performance, every time.** Re-run the equivalence check
  after any perf change.
- **Prefer tooling over one-off prompts.** When a manual AI workflow repeats
  (e.g. "diff these two PTX dumps for reduction order"), have the agent build a
  script for it. This is an explicit project expectation.
- **Point the agent at this KB.** When starting a fresh session on a sub-task,
  reference the relevant `knowledge-base/*.md` so it loads prior context instead
  of re-deriving it.
- **Capture dead ends too.** "We tried X, it didn't work because Y" saves the
  next session from repeating it.

## 7. Quick checklist per session

- [ ] Relevant `knowledge-base/*.md` referenced at start?
- [ ] Constraints stated in prompt + comments?
- [ ] Correctness re-verified after any perf-affecting change?
- [ ] New reusable concept → knowledge-base doc?
- [ ] CLAUDE.md still accurate (goals, infra, open questions)?
- [ ] "Summary today's work" written to `claude-chatting/`?
- [ ] Changes committed to git?
