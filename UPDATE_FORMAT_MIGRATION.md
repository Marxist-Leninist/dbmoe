# Safetensors-only worker updates

This change removes pickle decoding from the untrusted POST /update endpoint.
It does not change model math, update validation, gating, checkpoint files, or trusted coordinator lease responses.

## Two-stage rollout

1. Deploy the first commit (Python worker serializer and Content-Type change) to workers while coordinators still accept both formats. Current main already supports safetensors updates. Restart workers through the normal approved process and confirm successful accepted updates from each supported client type.
2. After active Python workers and downloadable worker bundles are updated, deploy the strict coordinator commit. It accepts only application/vnd.dbmoe.update+safetensors, with optional MIME parameters. Unsupported or missing types receive HTTP 415 before the body is read, and the connection closes.

There is no silent pickle fallback. Pre-safetensors coordinators cannot accept the new Python worker. Old Python workers cannot contribute after strict cutover; their existing retry loop may repeat HTTP 415 until upgraded. Plan the upgrade before cutover and announce the required client version.

The Rust worker already uses this MIME type and safetensors codec. Its actual binary loopback needs a built Rust worker; source compatibility alone is not a new hardware validation. Other third-party clients need explicit inventory and confirmation before rollout.

Do not deploy the full strict branch to a shared coordinator/worker source path just to upgrade workers: an unrelated coordinator restart could then enable strict mode early. Use a separate worker release path or the first commit only until cutover is approved.

## Rollback

If the coordinator is temporarily rolled back to dual-format support, new safetensors workers still work. This reopens the legacy decoding exposure, so treat it as a temporary operational rollback rather than the security endpoint.
Rolling workers back to pickle while the strict coordinator remains active breaks their updates.

## Scope and remaining limits

This removes the pickle deserializer from untrusted update ingestion; it does not claim complete denial-of-service protection.
The existing safetensors decoder checks dtype, dimensions, tensor count, 1 MiB header bound, exact byte lengths and nonoverlapping complete data spans before copying tensor data.
The existing total upload cap and concurrency settings remain unchanged. Their aggregate memory budget, connection limits and request deadlines are separate resource-hardening work.
Trusted torch checkpoint/lease loading remains unchanged.

## Isolated validation

Stage 1 was tested with the unchanged dual-format coordinator: 61 tests completed, 58 passed and three hardware-dependent skips.
The strict stage includes normal Python training loopbacks, existing safetensors loopback, scalar/tensor metadata roundtrip, HTTP header assertion, and early 415 checks using normal unsupported Content-Types with no body sent.
No production deployment or restart is part of this preparation.
