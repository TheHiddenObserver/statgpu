# Code Review Target Resolution

Resolve the review target before reading findings. The goal is to make a clean/blocking verdict reproducible and stale-detectable.

## Required report identity

Every review records:

- `target_kind`: `pr`, `branch`, `range`, `path`, or `working-tree`;
- `base_sha`: immutable comparison base when applicable;
- `head_sha`: immutable reviewed commit head;
- `default_branch`: the resolved repository default branch when used;
- `path_filter`: explicit path restriction, if any;
- `working_tree`: `clean` or `dirty`, plus changed/untracked paths when the working tree is part of scope;
- `resolution_command/source`: enough detail to reproduce how the target was resolved.

A PR/branch review is not identified by a branch name alone. Record immutable SHAs.

## Resolution rules

### Explicit PR (`#136`, `PR 136`, or a PR URL)

1. Resolve PR metadata and exact base/head SHAs. Prefer `gh pr view <N> --json baseRefName,headRefName,baseRefOid,headRefOid,url` when `gh` is available.
2. Ensure the base/head commits are available locally if source inspection uses git; fetch the named refs when needed.
3. Audit the immutable PR diff (`base_sha...head_sha`) and read file contents from that head. Do not silently substitute the current checkout.
4. For `auto-fix`, edit only when the writable checkout/worktree is demonstrably at `head_sha` **and clean before the fix**. If the checkout is on another commit or already contains unrelated dirty/untracked state, stop with a target/write-state mismatch or use an explicitly prepared clean worktree; do not layer PR fixes onto ambiguous pre-existing changes.
5. Immediately before the verdict, resolve the PR again. If either the PR `base_sha` or `head_sha` changed, the prior diff identity is stale and must be re-reviewed.

If exact PR metadata cannot be resolved, do not issue a clean/blocking verdict. Ask for or report the missing target information instead of guessing.

### Explicit branch

1. Resolve the repository default branch from `origin/HEAD` or repository metadata; do not assume a name if it can be discovered.
2. Resolve both the branch head SHA and the default-branch head used to compute the comparison.
3. Use the merge-base with the default branch as `base_sha` unless the caller supplied another base.
4. Review `base_sha...head_sha` and record both SHAs plus the default-branch head used to derive the merge-base.
5. Before the verdict, re-resolve the branch head **and the comparison/default-branch head** and recompute the merge-base. If the effective `base_sha` or `head_sha` changed, the old verdict is stale.

### Explicit commit/range

Resolve every symbolic ref to immutable SHAs and record the exact range semantics (`A..B` versus `A...B`). Do not silently change range semantics. If the caller supplied only immutable commit SHAs, later branch movement does not alter that range.

### Explicit path

A path is a filter, not a base/head definition. Apply it to an explicit PR/branch/range when one is supplied; otherwise apply it to the no-scope working-tree rule below.

### No explicit scope

Review the current branch's work relative to the repository default branch:

1. resolve the current `HEAD` SHA and the default-branch head;
2. resolve `base_sha = merge-base(HEAD, default-branch-head)`;
3. include committed branch changes from `base_sha...HEAD`;
4. include staged and unstaged tracked changes;
5. enumerate untracked paths and inspect all potentially task-relevant untracked files; any exclusions (for example ignored/generated artifacts) must be explicit rather than silently dropping untracked state;
6. record whether the working tree is clean/dirty and the changed/untracked paths;
7. before the verdict, re-resolve current `HEAD`, default-branch head/merge-base, and working-tree status. Unexpected changes to the effective base/head or audited dirty state make the prior review stale.

If the current branch is the default branch and has no divergent commits, the scope may consist only of working-tree changes.

## Audit versus auto-fix

`audit` must not mutate the target.

`auto-fix` records two identities:

- `reviewed_before`: the target/state that produced the findings;
- `reviewed_after`: the post-fix state that receives the final re-review.

For an explicit PR auto-fix, start from a clean worktree at the resolved PR head so `reviewed_after` contains only the intended fix state. If edits remain uncommitted, report the final `HEAD` plus dirty working-tree paths. If commits are separately authorized and created, report the resulting exact head SHA. Never cite CI or review evidence from an earlier head as proof of a later head.

## Fail-closed conditions

Do not return `REVIEW CLEAN`, approval, or another blocking-completion verdict when:

- PR/branch/range resolution is ambiguous;
- a requested PR cannot be resolved to exact SHAs;
- the effective comparison base or head moved during audit and was not re-reviewed;
- `auto-fix` would write to a checkout that is not the resolved target or is not clean before the fix;
- relevant untracked/dirty changes are known to exist but were excluded without an explicit path filter/exclusion rationale.
