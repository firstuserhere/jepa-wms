# Multi-agent worktrees

## Why this workflow

Git commits are immutable; branches are movable names. Use the annotated
onboarding tag as the common base, give each agent an isolated worktree and
branch, and integrate reviewed commits into the shared branch. This prevents
Cursor or other agents from overwriting each other's files while preserving a
reproducible starting point.

The canonical onboarding tag is `jepawm-droid-infra-2026-08-19`. It must never
be moved or recreated. Before work, resolve it to a full commit:

```bash
git fetch --all --tags --prune
git rev-parse 'jepawm-droid-infra-2026-08-19^{commit}'
git tag -v jepawm-droid-infra-2026-08-19 || git show --no-patch jepawm-droid-infra-2026-08-19
```

An annotated but unsigned tag may not pass `git tag -v`; `git show` still lets
you inspect the annotation and target. Record the resolved SHA in the task.

## Fresh coordinator clone

```bash
git clone https://github.com/firstuserhere/jepa-wms.git jepa-wms
cd jepa-wms
git remote rename origin fork
git remote add origin https://github.com/facebookresearch/jepa-wms.git
git fetch --all --tags --prune
```

This matches the original workstation's convention: `fork` is writable and
`origin` is Meta upstream. If a tool assumes `origin` is writable, keep the
default clone naming and add Meta as `upstream` instead; record the convention
in the task. Never guess which remote will receive a push.

## Create one worktree per agent

From the coordinator clone:

```bash
agent_name=alice
task_name=training-heartbeat
base_tag=jepawm-droid-infra-2026-08-19

git worktree add \
  "../jepa-wms-${agent_name}-${task_name}" \
  -b "codex/${agent_name}-${task_name}" \
  "$base_tag"
```

Open that new directory as the agent's Cursor workspace. Each agent owns:

- exactly one worktree;
- exactly one task branch;
- a narrow file/feature scope;
- its commits, tests, and handoff note.

Agents may read any worktree but must edit only their own. Shared generated
caches, datasets, and checkpoints live outside Git worktrees.

## Task handoff contract

Every agent should return:

```text
Goal:
Branch:
Base tag and resolved base SHA:
Head commit SHA:
Files changed:
Scientific behavior changed: yes/no, with details
Tests run and exact outcomes:
External actions taken: none/list
Known gaps or follow-ups:
```

No handoff should consist only of an uncommitted diff. The coordinator reviews
the commit and chooses one of:

```bash
# Integrate an independent, reviewed commit.
git switch codex/droid-dinov3-research-infra
git cherry-pick FULL_AGENT_COMMIT_SHA

# Or merge a multi-commit branch while preserving its history.
git merge --no-ff codex/alice-training-heartbeat
```

Resolve conflicts in the integration worktree, rerun combined tests, and never
ask one agent to solve conflicts inside another agent's worktree.

## Parallelization boundaries

Good independent worktree tasks:

- training-v1 heartbeat/MFU instrumentation;
- Pantheon renderer contract update;
- qualification result/receipt analysis;
- checkpoint storage stress tests;
- architecture experiments in new configs;
- documentation or test expansion in non-overlapping files.

Poor parallel boundaries:

- two agents editing the main training loop;
- two agents changing the same YAML overlay or launch wrapper;
- one agent changing checkpoint schema while another consumes it without an
  agreed interface;
- multiple agents launching or cancelling jobs independently.

For overlapping work, designate one interface owner first. Other agents write
tests or adapters against the agreed interface on separate branches.

## Updating the baseline

Do not move the 2026-08-19 tag. When a new stable base is earned:

1. integrate and test all intended commits;
2. update `docs/RESEARCH_STATUS.md` and `docs/DECISIONS.md`;
3. create a new dated annotated tag;
4. push the commit and tag;
5. tell agents to branch from the new tag explicitly.

Old worktrees remain reproducible because their base tag and SHA do not change.

## Cleanup

Only after the branch is pushed or integrated and the worktree has no unique
changes:

```bash
git worktree list
git -C ../jepa-wms-alice-training-heartbeat status --short
git worktree remove ../jepa-wms-alice-training-heartbeat
git worktree prune
```

Never use forced worktree removal as a routine cleanup step.
