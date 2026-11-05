# DBMoE web pages

## Live pages

- [Training progress](https://codex.opentransformers.online/dbmoe/progress)
- [Browser training self-test](https://codex.opentransformers.online/dbmoe/browser-selftest/)
- [Native worker joining instructions](https://codex.opentransformers.online/dbmoe/)

The progress dashboard reads the coordinator's public status and displays accepted work, evaluation history, network inference, and available self-reported worker hardware. Its inference controls offer measured automatic routing, a single coordinator, strict distributed inference without fallback, and the existing network-with-fallback mode. Automatic mode uses one fixed-public-text byte per route on the pinned snapshot, caches the faster result for that snapshot/sampler, and reports the selected route and probe timings.

## Browser test scope

The browser page runs two synthetic PyTorch reference fixtures with 2,945 parameters each. It compares forward results, gradients, and one AdamW update per fixture. The selected backend must initialize explicitly; there is no silent backend fallback. The test runs only after the user starts it, uses a separate worker, and has a two-minute limit.

This bounded test is separate from contributing to the live model. Its computed results remain in the page; ordinary static downloads are still visible to the web host.

For this release, CPU JavaScript and single-threaded WebAssembly parity passed in the cloud browser on 5 November 2026. The Stop control also passed its browser check. WebGPU did not initialize in that environment, so no GPU parity pass is claimed. A fixture pass establishes these small numerical checks on the tested backend, not full-model capacity or sustained performance.

## Runtime provenance and release contents

The fourteen browser runtime files are ready to serve; no application build is required. The six files under `browser-selftest/vendor/` come from these exact npm packages:

- `@tensorflow/tfjs@4.22.0`
- `@tensorflow/tfjs-backend-webgpu@4.22.0`
- `@tensorflow/tfjs-backend-wasm@4.22.0`

`browser-selftest/runtime-download-spec.json` records each registry archive, its SHA-512 integrity value, and the exact files extracted from `package/dist/`. To rebuild vendor files, fetch those archives, verify their recorded integrity, and extract only the listed files. The TensorFlow.js license is included. Runtime files are loaded from the same site.

`SHA256SUMS` covers the fourteen browser runtime files and the progress HTML snapshot. Verify them from this directory:

```sh
sha256sum -c SHA256SUMS
```

The runtime download specification is release metadata. It is not included in the public nginx asset allowlist.

## Deployment

`../deploy/browser-selftest.locations.conf` and `../deploy/browser-selftest.headers.conf` reproduce the narrow static routes and response headers. They contain no coordinator proxy, authentication data, or owner mapping.

The example uses the existing deployment directory `/opt/agillm43_netserve/dbmoe-browser-selftest/`. Copy only the fourteen browser runtime files there. Use mode 0755 for static directories and 0644 for their files, or equivalent permissions that let nginx traverse the directories and read the assets. Install the headers file at the exact path referenced by the location file, `/opt/agillm43_netserve/dbmoe-browser-selftest.headers.conf`, and include the location file inside the intended nginx server block. Adjust these filesystem paths together for another host.

The location allowlist serves only the named HTML, JavaScript, CSS, fixture, license, and WASM files. Other paths return 404. Check the full nginx configuration before reloading it.

Serve `progress/index.html` at `/dbmoe/progress` with the coordinator's status available at the same-origin `/dbmoe/status` endpoint.

## Native workers: one contributor across machines

After installing the native worker as described in the repository README, mint one reusable credential:

```sh
python3 dbmoe.py enrol \
  --url https://codex.opentransformers.online/dbmoe \
  --token-file worker.token
```

Copy that credential securely to each authorized machine. Give each worker a distinct name and use the same credential file:

```sh
python3 dbmoe.py worker \
  --url https://codex.opentransformers.online/dbmoe \
  --token @worker.token --name machine-1
```

The enrol command creates the credential file privately and does not print the token. Keep it out of source control, URLs, screenshots, and public messages. Reusing the credential gives machines one contributor identity while their distinct names keep worker records separate.

Anonymous automatic joining remains available: omit `--token` and the worker enrolls itself without an account. Workers have no default runtime cutoff (`--hours 0`). Self-enrolled credentials renew their 30-day inactivity horizon on authenticated use, so ongoing training has no lifetime expiry.
