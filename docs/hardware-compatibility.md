# Community hardware compatibility reports

Volunteer reports help identify failures on hardware the maintainers do not own. Participation is optional. A report records what its author says happened on one configuration, **not an independently verified support claim**.

- [View reports, most recently updated first](https://github.com/Marxist-Leninist/dbmoe/issues?q=is%3Aissue+in%3Atitle+%22%5BHardware+report%5D%22&s=updated&o=desc).
- [Open the voluntary hardware report form](https://github.com/Marxist-Leninist/dbmoe/issues/new?template=hardware-compatibility.yml).

This is a manual GitHub issue tracker. There is no telemetry server, automatic upload, background reporter, device registration, identity database or automatic follow-up messaging. The command does not join the training network. Normal workers do not opt into this reporting feature. Existing worker hardware-sharing settings are separate.

## Generate and review a local report

Use the current repository version of `dbmoe.py`; a separately deployed `/code` copy may not include this command yet. Install the correct PyTorch build using the README instructions, then run one small synthetic test:

```bash
python dbmoe.py hardware-report --device xpu --output hardware-report.json
```

Replace `xpu` with `cpu`, `cuda`, `rocm`, `mps`, or an indexed device where supported. `auto` records the selected backend; it does not prove a different GPU was tested. To deliberately include GPU model and total memory:

```bash
python dbmoe.py hardware-report --device xpu --include-hardware --xpu-memory-fraction 0.5 --output hardware-report-with-model.json
```

The XPU budget is optional and has the allocator-only limits described in the README. Other backends must omit it. The command uses local compute and memory, does not read a worker token, and makes no coordinator or GitHub request. It writes a new JSON file and refuses to overwrite an existing file or symlink. Without `--output`, JSON goes to standard output. A failed test still emits its safe report and exits with status 1. If Python or PyTorch cannot start, no report can be generated; choose “could not run” in the issue form instead.

Review the JSON yourself. Posting it is a separate decision and action. Do not paste the entire terminal session: framework warnings, raw tracebacks and other logs are not part of the safe report and can contain local paths or private details.

## Exactly what the report contains

Always included:

- Schema, DBMoE version and SHA-256 hash of the local script. Script contents and its path are not included.
- A fresh random report ID for this one run and UTC time to the minute. The ID is not stored for reuse and is not a machine identifier.
- Requested/selected device and backend, a small OS-family/architecture allowlist, and numeric PyTorch/runtime release versions. Local build suffixes and arbitrary platform strings are discarded.
- Test outcome and stage. Failures use fixed categories, never exception text, tracebacks, environment variables or command lines.
- Explicit self-reported/unverified status. The tiny test does not test full-model memory capacity or real distributed training. Passing checks cover forward/backward, AdamW, finite bounded CPU deltas and serialization.
- Configured XPU allocator fraction, if requested and the check completed. The absolute GiB budget is included only with hardware opt-in because it can reveal device capacity.

Only with `--include-hardware`: GPU model, rounded total device memory and a shared/dedicated/unknown classification when available. GPU names come from the driver, not serial or PCI inventory dumps; suspicious multiline/path/email-shaped names are withheld. An unavailable backend may have no hardware details. Total memory is not free memory, and integrated GPUs may share it with the CPU. `hardware_opt_in` records the local option, not consent to upload.

The code does not collect hostnames, IP addresses, device serials, persistent machine IDs, account names, emails, tokens, paths, prompts, datasets or training content. Optional model/version details can still make a setup recognizable. Review them before sharing and remove optional fields you do not want to publish. Do not add raw diagnostic dumps to replace withheld fields.

## Publish only if you choose to

1. Search existing reports to avoid duplicates.
2. Open the form and paste only the JSON you reviewed. Optional setup details should be limited to public GPU model, OS release and driver version.
3. Read the public-sharing notice and consent checkboxes, then submit yourself.

GitHub associates the public issue with your GitHub account. This software neither creates nor posts an issue for you. GitHub operates the website and necessarily receives normal connection data when you visit/post; the DBMoE report itself contains no IP address. GitHub's account, privacy and notification controls apply.

Opt out by never submitting, omitting `--include-hardware`, or stopping future reports. There is no reporting daemon to disable. You may delete your local JSON. For an already-public issue, use GitHub's editing/removal controls or contact the repository maintainer; copies, notifications, search indexes and retained records may persist. Closing an issue does not erase its contents. Do not publish anything that must remain secret.

## Keep failures and later results together

Use one issue per setup/problem and add a newly reviewed report after a meaningful change, such as a driver/PyTorch upgrade or a fix. State whether the problem remains and what changed. Do not continuously run the check or post repeated unchanged reports. Fresh report IDs distinguish attempts without persistently identifying the machine.

The issue list sorted by recent updates shows the latest volunteered findings. This feature does **not** automatically watch issues or promise response times. Future monitoring or automatic contact must be separately scoped and authorized. Report authors control whether to return and whether to enable GitHub notifications.

Maintainers should distinguish:

- **Reported failure:** a volunteer reports a problem; stage and exact build matter.
- **Reported pass:** a volunteer reports success for the stated test scope only.
- **Independently reproduced:** a maintainer actually repeats the test on identified hardware/OS, with separate evidence and explicit scope. Submitted JSON, screenshots or issue checkboxes cannot grant this status by themselves.
- **Resolved for this reporter:** a later volunteer test says the problem is fixed, not that all hardware with that name is certified.

Do not copy identifying or sensitive volunteer details into release notes or compatibility tables. Summaries should use minimum hardware/OS/version/outcome information and link the voluntarily public issue. This change does not invent or prefill any volunteer hardware validation.
