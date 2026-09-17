# DyPho-Style Figure Availability Audit

The current quantitative package deliberately builds Tables I-III only. The
locked figure layouts require real panel evidence and cannot be filled with
table values, text placeholders, or mislabeled local variants.

| Asset | Status | Available now | Blocking gap |
| --- | --- | --- | --- |
| Fig. 1 | Blocked | RGB-D, local trajectories/maps, DYN-19 novel-view renders | Constructed-feature panels and a label-compatible real baseline row |
| Fig. 2 | Blocked | Dynamic masks, RGB-D input, Gaussian map/checkpoint | Keypoint, feature-map/bin, match, and temporal-prior exports |
| Fig. 3 | Inputs ready | Four TUM local trajectories and public ground truth | Explicit associated and SE(3)-aligned plot trajectories must be exported upstream |
| Fig. 4 | Historical package complete; current-method comparison blocked | Protocol-300 contains 20 real 640x480 panels: input, SplaTAM, Photo-SLAM, mask-only, and historical failed-boundary Ours; DYN-19 has separate local variant renders | A co-registered DYN-19 Full cell against the Protocol-300 baselines |

The Protocol-300 qualitative package is real and complete, but its `Ours` row
is a failed-boundary historical ablation. It cannot be relabeled as DYN-19
Full. A current-method Fig. 4 therefore still requires a new DYN-19 Full run or
render under that frozen baseline package. DyPho-SLAM remains table-only
because no real local or source-documented render panel is registered for it.
