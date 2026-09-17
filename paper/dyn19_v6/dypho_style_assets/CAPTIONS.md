# Suggested Captions

**Table I.** Camera-tracking ATE RMSE on the four dynamic TUM RGB-D scenes
used by the DyPho-SLAM Table I layout. DYN-19 values are local means and
population standard deviations across seeds 0/1/2. The DyPho-SLAM row marked
`[ext]` is copied from the published reference table and is not a
protocol-matched local rerun. Lower is better.

**Table II.** DYN-19 common-view static-region mapping diagnostic over four
scenes, three seeds, and 20 retained non-keyframe views per scene-seed cell.
Each row averages 12 cells for online and GT-aligned rendering. Higher is
better. GT-aligned rendering remains an end-to-end reconstruction diagnostic
because online poses constructed the map. DyPho-SLAM reports no corresponding
numeric PSNR/SSIM mapping table in the checked source.

**Table III.** Local DYN-19 application end-to-end runtime, equal-weight mean
per-run failure rate, relative runtime, and recovered-support route
certificate across 30 runs per configuration. All runs used host `slam`, GPU
identifier 1, and a serial GPU lock; the frozen manifests do not record the
GPU model. No tracking/mapping runtime split is inferred.
