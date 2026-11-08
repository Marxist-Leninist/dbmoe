# Inference placement and replica recovery

DBMoE's inference worker now checks the snapshot and block in a downloaded bundle against the coordinator's assignment before constructing a replacement. Missing, boolean, string, fractional and mismatched identifiers are rejected. A failed load leaves the previous held block intact.

On the explicit `distributed` route, a volunteer whose initial audited result disagrees with the pinned model no longer makes a healthy replica unusable. The router discards that result, records the existing strike, and tries another worker for the same snapshot, block and numerical inputs. Each attempt has a fresh request ID. Retries share the existing finite hop retry budget. Exhaustion still fails closed: there is no coordinator substitution. The ordinary `network` route retains its existing explicitly reported coordinator fallback.

This is placement consistency and bounded recovery, not signed attestation or proof that every participant is honest. Initial open-worker hops are synchronously audited; later results retain the existing probabilistic asynchronous audit policy. A model owner can still supply a wrong bundle with matching identifiers. Transport security, result audits and credential controls remain necessary. Anonymous self-enrolment is unchanged.

## Evidence

`python -m unittest test_inference_integrity test_routing_v4 -q` exercises real local polling/serving with tiny CPU models, corrupt first-choice owners and correct replicas. It checks exact generated-byte parity with single-host execution, unchanged retry inputs, different attempt IDs, assignment validation, rejected malformed IDs, failure without a substitute and a zero-retry budget. Existing routing tests cover snapshot pinning, slow/dead peers, stale results, and route reporting. These are local process/thread tests, not a new physical multi-host throughput benchmark or a proof of model quality.

## Relationship to Swarmlet

The architectural review of https://github.com/lotar/swarmlet highlighted useful principles: pinned placements, explicit owner-loss behavior, deterministic parity and precise evidence boundaries. This implementation was written independently for DBMoE's existing protocol. No Swarmlet source code or dependencies were imported.

Swarmlet uses the custom Swarmlet Community License 1.0, which is source-available, not OSI open source, and restricts commercial use by corporate-group revenue. It must not be treated as an unrestricted code dependency. See https://github.com/lotar/swarmlet/blob/main/LICENSE for its terms.

DBMoE already has snapshot content hashes, fastest-idle-worker routing, exact pinned-snapshot routing, bounded retries, explicit distributed/no-substitution semantics and Rust/Python parity tests. They are retained rather than reimplemented. Signed manifests would require a separate key/trust design; WAN per-expert routing, speculative decoding and low-RAM model changes require workload-specific evidence and are not included here.
