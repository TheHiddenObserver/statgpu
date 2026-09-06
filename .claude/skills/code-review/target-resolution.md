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
4. For `auto-fix`, only edit when the writable checkout/worktree is demonstrably the PR head (or an explicitly prepared worktree for it). If not, stop with a target/write-state mismatch rather than editing another branch.
5. Immediately before the verdict, resolve the PR again. If `head_sha` changed, the prior review is stale.

If exact PR metadata cannot be resolved, do not issue a clean/blocking verdict. Ask for or report the missing target information instead of guessing.

### Explicit branch

1. Resolve the repository default branch from `origin/HEAD` or repository metadata; do not assume a name if it can be discovered.
2. Resolve the branch head SHA.
3. Use the merge-base with the default branch as `base_sha` unless the caller supplied another base.
4. Review `base_sha...head_sha` and record both SHAs.
5. Re-resolve the branch head before the verdict; a moved head invalidates the old verdict.

### Explicit commit/range

Resolve every symbolic ref to immutable SHAs and record the exact range semantics (`A..B` versus `A...B`). Do not silently change range semantics.

### Explicit path

A path is a filter, not a base/head definition. Apply it to an explicit PR/branch/range when one is supplied; otherwise apply it to the no-scope working-tree rule below.

### No explicit scope

Review the current branch's work relative to the repository default branch:

1. resolve the current `HEAD` SHA and default branch;
2. resolve `base_sha = merge-base(HEAD, default-branch-head)`;
3. include committed branch changes from `base_sha...HEAD`;
4. include staged and unstaged tracked changes;
5. include relevant untracked files rather than silently ignoring them;
6. record whether the working tree is clean/dirty and the changed/untracked paths.

If the current branch is the default branch and has no divergent commits, the scope may consist only of working-tree changes.

## Audit versus auto-fix

`audit` must not mutate the target.

`auto-fix` records two identities:

- `reviewed_before`: the target/state that produced the findings;
- `reviewed_after`: the post-fix state that receives the final re-review.

If edits remain uncommitted, report the final `HEAD` plus dirty working-tree paths. If commits are separately authorized and created, report the resulting exact head SHA. Never cite CI or review evidence from an earlier head as proof of a later head.

## Fail-closed conditions

Do not return `REVIEW CLEAN`, approval, or another blocking-completion verdict when:

- PR/branch/range resolution is ambiguous;
- a requested PR cannot be resolved to exact SHAs;
- the target moved during audit and was not re-reviewed;
- `auto-fix` would write to a checkout that is not the resolved target;
- relevant untracked/dirty changes are known to exist but were excluded without an explicit path filter.
