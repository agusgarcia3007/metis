# The Meridian Edge Standard (MES) — Technical Specification v4

The Meridian Edge Standard defines how compact inference systems run on constrained hardware.
It was first drafted by the Orrery Foundation and is now maintained as an open specification.

## Components

A conforming Meridian system is built from four named components:

- **Lumen** — the resident reasoning core. Lumen has a memory budget of 512 MB, the largest of
  any single component. It is codenamed Falconer in the reference build.
- **Aster** — the retrieval plane. Aster has a memory budget of 192 MB and is codenamed Pelican.
- **Quill** — the tool dispatcher. Quill has a memory budget of 96 MB and is codenamed Sandpiper.
- **Tideway** — the control loop. Tideway has a memory budget of 64 MB and is codenamed Curlew.

The total resident budget across all four components must never exceed 900 MB. Any build that
exceeds this ceiling is non-conforming and may not display the Meridian seal.

## Operating limits

- A conforming system may keep at most **5 retrieval segments** resident at any one time.
- Each retrieval segment is capped at **128 kilobytes** of decompressed text.
- The decode loop must sustain at least **12 tokens per second** on the reference hardware to be
  certified. Builds below this floor are graded "provisional" rather than "certified".
- The reference hardware is a **6 GB RAM, 4 vCPU** machine with no accelerator.

## Provenance rule

Every factual claim emitted by a Meridian system must carry a **lineage token** identifying the
retrieval segment it was drawn from. A claim without a lineage token is treated as unverified and
must be withheld from the user. This rule is the heart of the standard: a Meridian system is
permitted to say nothing, but is never permitted to say something it cannot trace.

## Conformance tiers

There are three conformance tiers, in increasing order of strictness: **Bronze**, **Cobalt**, and
**Vermillion**. A Vermillion build must pass every Cobalt requirement plus an independent audit of
its lineage tokens. Only Vermillion builds may be deployed in regulated environments.
