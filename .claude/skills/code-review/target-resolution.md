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
- `worktree_fingerprint`: required when staged/unstaged/untracked content is part of scope; use a content hash, not only a path list;
- `resolution_command/source`: enough detail to reproduce how the target was resolved.

A PR/branch review is not identified by a branch name alone. Record immutable SHAs.

## Working-tree fingerprint

When the working tree is part of the target, compute a deterministic fingerprint over the **content being reviewed**. A suitable contract is SHA-256 over a canonical byte stream containing, in order:

1. the current `HEAD` SHA;
2. `git diff --binary` for included unstaged tracked changes;
3. `git diff --cached --binary` for included staged changes;
4. each included untracked path in sorted path order plus a SHA-256 of that file's bytes.

Apply an explicit path filter consistently to the tracked diffs and untracked-file set. Record excluded generated/ignored artifacts rather than silently varying the fingerprint scope.

The exact helper/command may vary by platform, but the report must provide enough detail to recompute the same fingerprint. Rechecking only `git status` or only the changed path names is insufficient because file contents can change while the path set remains identical.

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
5. For `auto-fix` of this **committed branch target**, require a clean writable checkout at `head_sha` before editing; otherwise the target would silently expand to include unrelated working-tree changes.
6. Before the verdict, re-resolve the branch head **and the comparison/default-branch head** and recompute the merge-base. If the effective `base_sha` or `head_sha` changed, the old verdict is stale.

### Explicit commit/range

Resolve every symbolic ref to immutable SHAs and record the exact range semantics (`A..B` versus `A...B`). Do not silently change range semantics. If the caller supplied only immutable commit SHAs, later branch movement does not alter that range.

For `auto-fix`, an explicit commit/range is a read-only identity unless the caller also identifies a writable branch/worktree at the range head. Never guess which branch should receive fixes.

### Explicit path

A path is a filter, not a base/head definition. Apply it to an explicit PR/branch/range when one is supplied; otherwise apply it to the no-scope working-tree rule below. In `auto-fix`, do not modify outside the explicit path filter unless a newly discovered cross-file dependency is reported and the caller's requested scope clearly authorizes it.

### No explicit scope / working-tree review

Review the current branch's work relative to the repository default branch:

1. resolve the current `HEAD` SHA and the default-branch head;
2. resolve `base_sha = merge-base(HEAD, default-branch-head)`;
3. include committed branch changes from `base_sha...HEAD`;
4. include staged and unstaged tracked changes;
5. enumerate untracked paths and inspect all potentially task-relevant untracked files; any exclusions (for example ignored/generated artifacts) must be explicit rather than silently dropping untracked state;
6. record whether the working tree is clean/dirty, the changed/untracked paths, and a `worktree_fingerprint` over the included content;
7. before the verdict, re-resolve current `HEAD`, default-branch head/merge-base, working-tree status, and the fingerprint. Any unexpected change to the effective base/head or audited content fingerprint makes the prior review stale.

A dirty working tree is **expected and valid** for no-scope development review: it is part of `reviewed_before`, not a reason to refuse auto-fix. Snapshot the pre-fix paths and fingerprint before editing so later changes can be attributed to the review/fix pass.

If the current branch is the default branch and has no divergent commits, the scope may consist only of working-tree changes.

## Audit versus auto-fix

`audit` must not mutate the target.

`auto-fix` records two identities:

- `reviewed_before`: the target/state that produced the findings, including its fingerprint when dirty;
- `reviewed_after`: the post-fix state that receives the final re-review, with a new fingerprint when dirty.

For an **explicit committed target** (PR or branch), start from a clean worktree at the resolved head. For a **no-scope working-tree target**, preserve the captured dirty state as the legitimate pre-fix target and distinguish it from review-created edits. If edits remain uncommitted, report the final `HEAD` plus dirty working-tree paths/fingerprint. If commits are separately authorized and created, report the resulting exact head SHA. Never cite CI or review evidence from an earlier head/fingerprint as proof of a later state.

## PR comment freshness

When `--comment` is requested, the published review summary must correspond to the PR's currently resolved remote `base_sha`/`head_sha`.

If `auto-fix` produced only local uncommitted or unpushed changes, a PR comment may describe the **original PR findings** and clearly label the fixes as local draft work, but it must not claim the remote PR is fixed/clean. A post-fix clean verdict may be posted only after any separately authorized push/update is visible in the PR and the PR is re-resolved/re-reviewed at that new remote head.

## Fail-closed conditions

Do not return `REVIEW CLEAN`, approval, or another blocking-completion verdict when:

- PR/branch/range resolution is ambiguous;
- a requested PR cannot be resolved to exact SHAs;
- the effective comparison base or head moved during audit and was not re-reviewed;
- a working-tree fingerprint changed unexpectedly during audit and the new content was not re-reviewed;
- `auto-fix` of an explicit committed target would write to a checkout that is not the resolved target or is not clean before the fix;
- relevant untracked/dirty changes in a working-tree review are known to exist but were excluded without an explicit path filter/exclusion rationale;
- a PR comment would describe an unpushed local fix state as though it were the current remote PR head.
