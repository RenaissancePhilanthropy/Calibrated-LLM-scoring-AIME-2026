# Colours shared by every figure, so a method looks the same in all four.
# Okabe-Ito, as used in the paper's Figure 1. Sourced by paper_figs.R and
# interp_figs.R; run those from the package root.

method_cols <- c("Judge" = "#E69F00", "Direct" = "#0072B2", "Channel" = "#009E73")
method_shapes <- c("Judge" = 16, "Direct" = 17, "Channel" = 15)

# Verdict shading in the interpretability figure: above / below the 0.5
# decision line, and the sign of a per-token LLR bar. Same palette, but a
# different meaning from the method colours above, so named separately.
OK_FILL <- "#009E73"
BAD_FILL <- "#D55E00"
