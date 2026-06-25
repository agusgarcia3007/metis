# Meridian Edge Standard — Version History

This document records the changes between ratified versions of the Meridian Edge Standard.

## Version 4 (ratified 2034)

- Raised the Lumen memory budget from 384 MB to its current 512 MB.
- Introduced the Vermillion conformance tier and the independent lineage-token audit.
- Lowered the decode floor from 15 tokens per second to the current 12 tokens per second, after
  field data showed the higher floor excluded too many low-power devices.
- Added the requirement that every claim carry a lineage token. In version 3, provenance was
  recommended but not mandatory; version 4 made it mandatory.

## Version 3 (ratified 2029)

- Split the original monolithic "Engine" component into the separate Lumen and Aster components.
- Established the four-working-group structure still in use today.
- Set the total resident budget ceiling at 900 MB, unchanged since.

## Version 2 (ratified 2026)

- First version to define conformance tiers (Bronze and Cobalt only; Vermillion came later).
- Introduced the reference-hardware definition.

## Version 1 (ratified 2024)

- The original draft. Defined a single Engine component and no formal limits.

## Deprecations

The "Engine" component name was deprecated in version 3 and must not appear in any conforming
version 4 build. Builds that still expose an "Engine" component are rejected at audit.
