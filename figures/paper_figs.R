# Three SciEntsBank paper figures, Direct = pq_redacted.
# Colours come from figures/palette.R, shared with interp_figs.R.
#
# Inputs (written by export/export_figure_data.py):
#   export/fig1_data.csv
#   figures/figdata/channel_9b.csv, figures/figdata/judge_9b.csv, figures/figdata/direct_9b.csv
#     (direct_9b = the pq_redacted per-sample cell)
# Run from the package root:  Rscript figures/paper_figs.R
# Outputs: figures/generated/scientsbank_2way_methods_final.{pdf,png}
#          figures/generated/reliability_scientsbank_final.{pdf,png}
#          figures/generated/coverage_at_accuracy_scientsbank_final.{pdf,png}

library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)

source("figures/palette.R")

dir.create("figures/generated", showWarnings = FALSE, recursive = TRUE)

# ============================ Figure 1 =====================================
raw <- read.csv("export/fig1_data.csv", stringsAsFactors = FALSE)

family_levels <- c("Qwen3.5", "Llama-3.2", "OpenAI")
model_top2bottom <- c(
  "Qwen3.5-9B (think)", "Qwen3.5-9B (no think)",
  "Qwen3.5-4B (think)", "Qwen3.5-4B (no think)",
  "Qwen3.5-2B (no think)", "Qwen3.5-0.8B (no think)",
  "Llama-3.2-3B", "Llama-3.2-1B",
  "gpt-5.5 (think)", "gpt-5.5 (no think)", "gpt-4o", "gpt-4o-mini"
)
model_levels <- rev(model_top2bottom)

DIRECT_VARIANT <- "pq_redacted"   # the direct method shown in all figures

fig1 <- raw %>%
  filter(method != "direct" | variant == DIRECT_VARIANT) %>%
  mutate(series = factor(
    c(judge = "Judge", direct = "Direct", channel = "Channel")[method],
    levels = c("Judge", "Direct", "Channel")))

long <- fig1 %>%
  pivot_longer(cols = c(acc, ece), names_to = "metric", values_to = "value") %>%
  filter(!is.na(value)) %>%
  mutate(
    model  = factor(model, levels = model_levels),
    family = factor(family, levels = family_levels),
    metric = factor(metric, levels = c("acc", "ece"),
                    labels = c("Accuracy", "ECE"))
  )

span <- long %>%
  group_by(family, model, metric) %>%
  summarise(lo = min(value), hi = max(value), n = n(), .groups = "drop") %>%
  filter(n > 1)

base_theme <- theme_minimal_hgrid(font_size = 12) +
  theme(
    panel.grid.major.x = element_line(colour = "grey90"),
    panel.grid.major.y = element_blank(),
    panel.spacing.y    = unit(10, "pt"),
    strip.text.y.left  = element_text(angle = 0, face = "bold", hjust = 0),
    strip.placement    = "outside",
    plot.title         = element_text(size = 12),
    plot.title.position = "panel"
  )

make_panel <- function(metric_name, panel_title, xlimits, xbreaks,
                       ref = NULL, ref_lab = NULL, show_y = TRUE) {
  d  <- filter(long, metric == metric_name)
  sp <- filter(span, metric == metric_name)
  p <- ggplot(d, aes(value, model)) +
    facet_grid(family ~ ., scales = "free_y", space = "free_y", switch = "y")
  if (!is.null(ref)) {
    p <- p + geom_vline(xintercept = ref, linetype = "dashed",
                        colour = "grey55", linewidth = 0.4)
  }
  p <- p +
    geom_segment(data = sp, aes(x = lo, xend = hi, y = model, yend = model),
                 inherit.aes = FALSE, colour = "grey75", linewidth = 1.1) +
    geom_point(aes(colour = series, shape = series), size = 3) +
    scale_colour_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(limits = xlimits, breaks = xbreaks,
                       expand = expansion(mult = 0.02)) +
    labs(title = panel_title, x = NULL, y = NULL) +
    base_theme
  if (!is.null(ref) && !is.null(ref_lab)) {
    p <- p + annotate("text", x = ref, y = Inf, label = ref_lab,
                      vjust = 1.3, hjust = -0.05, size = 3, colour = "grey55")
  }
  if (!show_y) {
    p <- p + theme(axis.text.y = element_blank(),
                   strip.text.y.left = element_blank())
  }
  p
}

p_acc <- make_panel("Accuracy", "Accuracy  (higher is better)",
                    c(0.5, 0.88), seq(0.5, 0.85, 0.1),
                    ref = 0.5, ref_lab = "chance", show_y = TRUE)
p_ece <- make_panel("ECE", "ECE  (lower is better)",
                    c(0, 0.55), seq(0, 0.5, 0.1),
                    ref = 0, ref_lab = "ideal", show_y = FALSE)
legend <- cowplot::get_plot_component(
  p_acc + theme(legend.position = "bottom",
                legend.justification = "center",
                legend.title = element_blank()),
  "guide-box-bottom")
panels <- plot_grid(p_acc + theme(legend.position = "none"),
                    p_ece + theme(legend.position = "none"),
                    nrow = 1, rel_widths = c(1.45, 1), align = "h", axis = "tb")
title <- ggdraw() +
  draw_label("Scoring methods on 2-way SciEntsBank (test_ua, n = 540)",
             fontface = "bold", x = 0, hjust = 0, size = 14)
final1 <- plot_grid(title, panels, legend, ncol = 1,
                    rel_heights = c(0.06, 1, .06))
ggsave("figures/generated/scientsbank_2way_methods_final.pdf", final1, width = 9.5, height = 4)
ggsave("figures/generated/scientsbank_2way_methods_final.png", final1, width = 9.5, height = 4,
       dpi = 300, bg = "white")

# ============================ Figure 2: reliability ========================
# Per-sample positive-class calibration; Gaussian kernel (bw = 0.05)
# smoothed curve; sample-level bootstrap (n = 100) 95% band; curve/band
# masked where the effective kernel-weighted sample count < 1; bottom row:
# 60-bin histogram.
BW <- 0.05; NGRID <- 200; NBOOT <- 100; MIN_SUPPORT <- 1.0
grid_x <- seq(0, 1, length.out = NGRID)

kernel_smooth <- function(p, y, grid, bw) {
  sapply(grid, function(g) {
    w <- exp(-0.5 * ((p - g) / bw)^2)
    d <- sum(w)
    if (d < 1e-10) 0 else sum(w * y) / d
  })
}
kernel_support <- function(p, grid, bw) {
  sapply(grid, function(g) sum(exp(-0.5 * ((p - g) / bw)^2)))
}
smooth_with_band <- function(p, y, seed = 42) {
  curve <- kernel_smooth(p, y, grid_x, BW)
  set.seed(seed)
  boot <- replicate(NBOOT, {
    idx <- sample.int(length(p), replace = TRUE)
    kernel_smooth(p[idx], y[idx], grid_x, BW)
  })
  lower <- apply(boot, 1, quantile, 0.025)
  upper <- apply(boot, 1, quantile, 0.975)
  bad <- kernel_support(p, grid_x, BW) < MIN_SUPPORT
  curve[bad] <- NA; lower[bad] <- NA; upper[bad] <- NA
  tibble::tibble(x = grid_x, curve = curve, lower = lower, upper = upper)
}

panel_data <- list(
  Channel = read.csv("figures/figdata/channel_9b.csv"),
  Judge   = read.csv("figures/figdata/judge_9b.csv"),
  Direct  = read.csv("figures/figdata/direct_9b.csv")
)
panel_titles <- c(
  Channel = "Channel, Qwen3.5-9B-Base",
  Judge   = "Judge, Qwen3.5-9B (no think)",
  Direct  = "Direct, Qwen3.5-9B-Base"
)

rel_panels <- lapply(names(panel_data), function(m) {
  d <- panel_data[[m]]
  sm <- smooth_with_band(d$p_correct, d$y_correct)
  col <- method_cols[[m]]
  main <- ggplot(sm, aes(x, curve)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted",
                colour = "gray", linewidth = 0.4) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = col, alpha = 0.2) +
    geom_line(colour = col, linewidth = 0.7, na.rm = TRUE) +
    annotate("text", x = 0.05, y = 0.92, label = sprintf("n = %d", nrow(d)),
             hjust = 0, size = 3.5, colour = "gray40") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = panel_titles[[m]],
         y = if (m == "Channel") "Empirical fraction correct" else NULL, x = NULL) +
    theme_minimal_grid(font_size = 12) +
    theme(axis.text.x = element_blank(),
          plot.title = element_text(size = 12))
  # Top row shares one 0-1 y scale: tick labels only on the leftmost panel.
  if (m != "Channel") {
    main <- main + theme(axis.text.y = element_blank(),
                         axis.ticks.y = element_blank())
  }
  dens <- ggplot(d, aes(p_correct)) +
    geom_histogram(bins = 60, fill = col, alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(x = "Predicted P(correct)",
         y = if (m == "Channel") "Count" else NULL) +
    theme_minimal_grid(font_size = 12)
  list(main = main, dens = dens)
})

final2 <- plot_grid(
  rel_panels[[1]]$main, rel_panels[[2]]$main, rel_panels[[3]]$main,
  rel_panels[[1]]$dens, rel_panels[[2]]$dens, rel_panels[[3]]$dens,
  ncol = 3, rel_heights = c(3, 1.15), align = "v", axis = "lr"
)
ggsave("figures/generated/reliability_scientsbank_final.pdf", final2, width = 9.5, height = 4.4)
ggsave("figures/generated/reliability_scientsbank_final.png", final2, width = 9.5, height = 4.4,
       dpi = 300, bg = "white")

# ============ Figure 3: coverage at a required accuracy ====================
# The deployment question, read left to right: "if I require accuracy >= t on
# the auto-graded subset, what fraction of grading can be automated?"
# y = achievable coverage (higher is better everywhere). How far a curve
# extends to the RIGHT is confidence quality: a method whose confidence
# signal is uninformative collapses as soon as the bar exceeds its overall
# accuracy.
#
# Operating points: samples admitted in ascending prediction entropy, entropy
# tie-blocks admitted whole (keep the LAST point of each block). Points
# covering fewer than MIN_SAMPLES samples are suppressed, since subsets that
# small are dominated by individual-sample noise; this only trims the far
# right of the curves, where a handful of samples would otherwise hold a
# shelf of coverage up to a required accuracy of 1.0.
MIN_SAMPLES <- 50
coverage_accuracy <- function(entropy_bits, correct) {
  o <- order(entropy_bits)
  ent <- entropy_bits[o]; cor <- correct[o]
  n <- length(ent)
  cum_acc <- cumsum(cor) / seq_len(n)
  coverage <- seq_len(n) / n
  block_end <- c(ent[-1] != ent[-n], TRUE)
  keep <- block_end & (seq_len(n) >= MIN_SAMPLES)
  tibble::tibble(coverage = coverage[keep], accuracy = cum_acc[keep])
}
coverage_at_accuracy <- function(op, t_grid) {
  sapply(t_grid, function(t) {
    ok <- op$accuracy >= t
    if (any(ok)) max(op$coverage[ok]) else 0
  })
}

t_grid <- seq(0.72, 1.0, by = 0.0025)
flip <- bind_rows(lapply(names(panel_data), function(m) {
  d <- panel_data[[m]]
  op <- coverage_accuracy(d$entropy_bits, d$correct)
  tibble::tibble(
    t = t_grid,
    coverage = coverage_at_accuracy(op, t_grid),
    series = factor(m, levels = c("Channel", "Judge", "Direct"))
  )
}))

final3 <- ggplot(flip, aes(t, coverage, colour = series)) +
  geom_step(linewidth = 0.8, direction = "hv") +
  # Method names only (the model is named in the caption); the longer labels
  # collided with the curves at this canvas size.
  scale_colour_manual(values = method_cols, name = NULL) +
  scale_x_continuous(limits = c(0.72, 1.0),
                     breaks = seq(0.75, 1.0, 0.05)) +
  scale_y_continuous(limits = c(0, 1.02)) +
  labs(title = "Coverage at a required accuracy",
       x = "Required accuracy on auto-graded subset",
       y = "Fraction auto-gradable (coverage)") +
  theme_minimal_grid(font_size = 12) +
  theme(plot.title = element_text(size = 12),
        legend.position = c(0.98, 0.98),
        legend.justification = c(1, 1),
        legend.background = element_blank())
# Narrower than Figs 1-2 (9.5 in): this one is printed one column wide, and
# the smaller canvas gives it the same shrink factor, so text matches.
ggsave("figures/generated/coverage_at_accuracy_scientsbank_final.pdf", final3,
       width = 4.75, height = 3.0)
ggsave("figures/generated/coverage_at_accuracy_scientsbank_final.png", final3,
       width = 4.75, height = 3.0, dpi = 300, bg = "white")

# One-number callouts for the running text.
for (m in names(panel_data)) {
  d <- panel_data[[m]]
  op <- coverage_accuracy(d$entropy_bits, d$correct)
  for (t in c(0.90, 0.95)) {
    ok <- op$accuracy >= t
    cat(sprintf("%-8s coverage at >=%.0f%% accuracy: %s\n", m, 100 * t,
                if (any(ok)) sprintf("%.1f%%", 100 * max(op$coverage[ok])) else "0%"))
  }
}

cat("wrote figures/generated/scientsbank_2way_methods_final.{pdf,png},",
    "figures/generated/reliability_scientsbank_final.{pdf,png},",
    "figures/generated/coverage_at_accuracy_scientsbank_final.{pdf,png}\n")
