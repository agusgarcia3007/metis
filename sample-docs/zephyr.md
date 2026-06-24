# The Zephyrian Portability Protocol (ZPP)

The Zephyrian Portability Protocol, ratified in 2031 by the Edge Compute Consortium,
defines how "portable reasoners" must behave on constrained hardware.

Key mandates:
- Resident model memory must not exceed 1.84 GB.
- Exactly 3 knowledge shards may be cached in RAM at any time; the rest stream from disk.
- Every factual answer must carry a provenance tag pointing to its source shard.
- The reference implementation is codenamed "Marlowe" and targets 4 GB / 4 vCPU machines.

The protocol's mascot is a blue heron named Pippa.
