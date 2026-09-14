# DYN-19 Independent Truth Gate

Date: September 14, 2026

## Current State

DYN-19 currently computes an occluded-background proxy from RGB-D frames and
dataset poses. That proxy is not an independent ghost ground truth and cannot
by itself establish zero ghost contamination or complete background recovery.

The local independent-annotation packs contain only pending manifests:

- no approved manual dynamic masks;
- no approved external-GT masks;
- no adjudication result;
- no revealed-background mask with reviewer provenance;
- no independent ghost/completeness metric certificate.

Consequently, current DYN-19 proxy values remain diagnostic and must be
reported separately from ground-truth claims.

## Required Evidence

Before promoting ghost or background-completeness claims, the release must
contain one of these independently sourced paths:

1. blind manual masks from an annotator independent of method development,
   with reviewer identity, approval state, ignore masks, and a second-reviewer
   agreement/adjudication record; or
2. external ground-truth masks/reference views with source provenance,
   alignment rules, and a documented separation from every DYN-19 prediction
   mask.

For revealed-background evaluation, the held-out render manifest must also
identify static pixels that were occluded by a moving object and later
validated against an independent reference. A current-frame RGB comparison is
not sufficient because a foreground trace can score well against the moving
observation.

## Promotion Rule

Only rows with approved independent labels, complete render/depth artifacts,
and a passing provenance audit may enter a claim-bearing ghost or completeness
table. Pending labels, model-generated pseudo-labels, DYN-19 semantic masks,
and empty proxy support remain boundary or diagnostic evidence.
