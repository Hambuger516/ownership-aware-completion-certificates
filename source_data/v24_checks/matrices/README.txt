V24 mean-only matrices

Cells show geometric means of four matched process pairs at fixed 9 pt. Both 2x2 single-column figures share the equality-centred log2 ratio scale (1/64, 1, 4). Blue/ochre indicate below/above one. The adverse 1.031 baseline cell remains outlined.

endpoint_groups.json contains 60 endpoint groups. all_four_paired_ratios.csv retains all 240 original ratios. baselines_GM_observed_ranges.csv and scale14b_GM_observed_ranges.csv contain 18 and 12 rows respectively, with both endpoints, geometric means, observed minima/maxima and all four paired ratios. These observed ranges are not confidence intervals. Every group is independently recomputed from the public frozen measurements and equals the v23 plotted record.

Reproduce: python figure_source/redesign_main_matrices_v24.py --manuscript-root .
The public recipe emits figures and complete table data as CSV; manuscript TeX is excluded. It uses matplotlib, numpy, pypdf and installed Times New Roman fonts.
