# ADR 0002 — dataset digests exclude modification time

Date: 2026-09-22
Status: accepted
Relates to: DEVELOPMENT.md §8.4, §10.4, `recorder/dataset.py`

## Context

"Did the data change?" is the single most useful question LabBox answers, so the digest
that represents a dataset must not move for reasons unrelated to content.

An obvious first implementation hashes `path + size + mtime + sampled bytes`. Measured
behaviour of mtime:

* copying a dataset with `shutil.copytree`, unzipping an archive, or a fresh `git clone`
  rewrites every mtime;
* editing one file in place can leave mtime unchanged if a tool restores it;
* different filesystems store mtime with different resolutions.

Including it means `dataset.digest changed` fires after a routine copy — the false positive
that would teach users to ignore the most important diagnostic in the tool.

## Decision

```text
entries   = files under the dataset root, excluding patterns, sorted by POSIX rel path
per_file  = f"{rel_path}|{size_bytes}|{sha256(head 64B)}|{sha256(tail 64B)}"
digest    = "sha256:" + sha256("\n".join(per_file))
```

mtime is recorded per file as metadata only, and a suspicious mtime produces a separate
`dataset_files_touched` style diagnostic worded as "may have been edited in place".

`fast_manifest` is explicitly documented — in the manifest, in `diagnostics.json`, in the
HTML report and in `reproduce.*` — as **not** a content proof. Users who need that
guarantee opt into `strategy = "full_content"`.

## Consequences

* Two datasets with identical content at different absolute paths share a digest, because
  the paths are relative to the dataset root. That is intended.
* A file whose bytes change only in the middle, with size preserved, is invisible to
  `fast_manifest`. This is stated in the warnings rather than hidden.
* Truncation (file-count or time budget) is recorded with `truncated=true` and a warning
  that such a digest is not comparable with a complete one — never silently compared.
