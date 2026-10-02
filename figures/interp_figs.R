# Token-level interpretability: channel (native) vs direct and judge
# (prefix-reveal), on the paper's Fig-4 examples.
#
# Row layout, per example:
#   1  combined P(correct) for all three methods against the 0.5 decision
#      line (green region = graded correct, red = incorrect)
#   2  LLR channel   3  LLR direct   4  LLR judge
# All four share the same token x-axis, labelled once under the bottom panel.
# The LLR bars are per-token log-odds evidence (nats): unlike delta-P they
# stay visible where the probability curve saturates near 0/1.
#
# Inputs: figures/figdata/interp_<idx>_{channel,direct,judge}.csv
#         (from interpretability/prefix_reveal.py)
# Run from the package root:  Rscript figures/interp_figs.R
# Output: figures/generated/interp_prefix_reveal_stacked.{pdf,png}

library(ggplot2)
library(dplyr)
library(cowplot)

dir.create("figures/generated", showWarnings = FALSE, recursive = TRUE)

source("figures/palette.R")

# Sized for one printed column: the canvas is CANVAS_W inches wide and the
# fonts are chosen so that at ~3 in they land at ~7.5-8 pt effective. If you
# change CANVAS_W, scale the font constants with it.
CANVAS_W <- 4.6
CANVAS_H <- 8.5
FS_BASE  <- 10.5  # theme base (tick labels, y titles)
FS_TOKEN <- 8.5   # angled token labels
FS_HEAD  <- 9.5   # per-example Q/Ref header

# Panel order down the figure, independent of the palette's key order.
METHODS <- c("Channel", "Direct", "Judge")

SAMPLES <- list(
  list(idx = 491,
       title = paste("Q: One function of the bess beetle's elytra is",
                     "protection.\nWhat is another function?",
                     "\nRef: The elytra are used to make sounds.",
                     "")),
  list(idx = 229,
       title = paste("Q: What will happen to the motor if the light bulb",
                     "burns out?",
                     "\nRef: The motor will continue to run (still has a",
                     "pathway\nto the D-cell)."))
)

read_panel <- function(idx, method) {
  f <- sprintf("figures/figdata/interp_%d_%s.csv", idx, tolower(method))
  d <- read.csv(f, stringsAsFactors = FALSE)
  d$method <- method
  # Per-method bar semantics:
  #   Channel: the shipped t_llr — the true per-token likelihood ratio
  #            (P(correct) = sigmoid of its cumulative sum).
  #   Direct:  the shipped per-reveal change in logit P_cal(correct) — exact,
  #            its softmax probabilities never reach 0/1.
  #   Judge:   recomputed here with P clipped to [0.01, 0.99]: verbalized
  #            confidences of 0.0/1.0 are literal logit infinities; +-4.6
  #            nats renders a coarse verbalized certainty honestly without
  #            dwarfing the panel.
  # Every panel gets an explicit step-0 row showing the method's prior:
  # channel = uniform 0.5 (synthetic row); direct/judge = their own t=0
  # ([REDACTED]-only) evaluation (direct's is exactly 0.5 by construction).
  # Step-0 has no bar.
  if (method == "Channel") {
    d <- dplyr::bind_rows(
      data.frame(step = 0, token = "<null>", p_correct = 0.5, bar = NA,
                 method = method),
      d)
  }
  if (method == "Judge") {
    p <- pmin(pmax(d$p_correct, 0.01), 0.99)
    lo <- log(p / (1 - p))
    d$bar <- c(NA, diff(lo))
  } else {
    d$bar[1] <- NA
  }
  d$x <- seq_len(nrow(d))
  d
}

token_labels <- function(d) {
  # "Ċ" is the tokenizer's byte-encoding of the template's answer-terminating
  # newline (as "Ġ" is space) — label it as a return, not a period-lookalike.
  lab <- ifelse(is.na(d$token) | d$token == "<null>", "∅",
                ifelse(trimws(d$token) == "Ċ", "⏎", trimws(d$token)))
  make.unique(lab)
}

x_axis <- function(n, labels) {
  scale_x_continuous(breaks = seq_len(n), labels = labels,
                     limits = c(0.5, n + 0.5),
                     expand = expansion(mult = 0.01))
}

prob_panel <- function(ds, labels, show_legend) {
  d <- bind_rows(ds)
  d$method <- factor(d$method, levels = METHODS)
  ggplot(d, aes(x, p_correct, colour = method)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 1,
             fill = OK_FILL, alpha = 0.07) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.5,
             fill = BAD_FILL, alpha = 0.07) +
    geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey45",
               linewidth = 0.35) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 1.1) +
    scale_colour_manual(values = method_cols, name = NULL) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                       labels = function(x) sprintf("%.1f", x)) +
    x_axis(max(d$x), labels) +
    labs(x = NULL, y = "P(correct)") +
    theme_minimal_grid(font_size = FS_BASE) +
    theme(axis.text.x = element_blank(),
          legend.position = if (show_legend) c(0.99, 0.45) else "none",
          legend.justification = c(1, 0.5),
          legend.background = element_blank(),
          legend.text = element_text(size = FS_BASE - 1.5),
          legend.key.height = unit(0.55, "lines"))
}

llr_panel <- function(d, method, labels, show_tokens) {
  ggplot(d, aes(x, bar, fill = bar >= 0)) +
    geom_col(width = 0.75, show.legend = FALSE) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.3) +
    scale_fill_manual(values = c(`TRUE` = OK_FILL, `FALSE` = BAD_FILL)) +
    # Few, uniformly-formatted ticks: the panels are short, and equal tick
    # widths keep the panel left edges aligned down the stack. The range is
    # forced to cover at least [-0.5, 0.5] so a panel whose bars sit mostly
    # on one side of zero (judge) still shows a usable scale. Breaks: the
    # smallest nice step yielding at most ~3 in-range ticks (pretty(n = 3)
    # can return 4+, which crowds these short panels — e.g. 0/2/4/6 on the
    # judge's 0-6 panel where 0/3/6 fits).
    scale_y_continuous(limits = c(min(-0.5, min(d$bar, na.rm = TRUE)) * 1.05,
                                  max(0.5, max(d$bar, na.rm = TRUE)) * 1.05),
                       breaks = function(lims) {
                         # smallest nice step giving at most 3 in-range ticks
                         for (step in c(0.5, 1, 2, 3, 5, 10)) {
                           b <- seq(floor(lims[1] / step),
                                    ceiling(lims[2] / step)) * step
                           b <- b[b >= lims[1] & b <= lims[2]]
                           if (length(b) <= 3) return(b)
                         }
                         b
                       },
                       labels = function(x) sprintf("%.1f", x)) +
    x_axis(nrow(d), labels) +
    # Two-line title: rotated single-line "Channel LLR" is taller than these
    # short panels, so adjacent panels' titles collide; stacking the words
    # halves the rotated height.
    labs(x = NULL, y = paste0(method, "\nLLR")) +
    theme_minimal_grid(font_size = FS_BASE) +
    theme(
      # hjust = 0.5 pins the rotated title to the vertical centre of the
      # panel; without it the bottom panel's title drifts because its slot
      # also holds the angled token labels.
      axis.title.y = element_text(colour = method_cols[[method]],
                                  size = FS_BASE - 1, lineheight = 0.9,
                                  hjust = 0.5, vjust = 0.5),
      axis.text.x = if (show_tokens)
        element_text(angle = 55, hjust = 1, size = FS_TOKEN)
        else element_blank())
}

sections <- lapply(seq_along(SAMPLES), function(si) {
  s <- SAMPLES[[si]]
  methods <- METHODS
  ds <- setNames(lapply(methods, function(m) read_panel(s$idx, m)), methods)
  labels <- token_labels(ds$Channel)

  panels <- c(
    list(prob_panel(ds, labels, show_legend = TRUE)),
    lapply(seq_along(methods), function(i) {
      llr_panel(ds[[methods[i]]], methods[i], labels,
                show_tokens = (i == length(methods)))
    })
  )
  grid <- plot_grid(plotlist = panels, ncol = 1, align = "v", axis = "lr",
                    rel_heights = c(2.1, 1, 1, 1.9))
  header <- ggdraw() +
    draw_label(s$title, x = 0.02, hjust = 0, size = FS_HEAD,
               fontface = "italic", lineheight = 1.0, colour = "black")
  plot_grid(header, grid, ncol = 1, rel_heights = c(0.16, 1))
})

final <- plot_grid(plotlist = sections, ncol = 1)
title <- ggdraw() +
  draw_label(paste0("Token-level interpretability:\nchannel (native) vs ",
                    "direct and judge (prefix-reveal)"),
             fontface = "bold", x = 0.02, hjust = 0, size = FS_BASE + 1,
             lineheight = 1.0)
out <- plot_grid(title, final, ncol = 1, rel_heights = c(0.045, 1))
# cairo_pdf: R's default pdf device has no glyphs for ⏎/∅; cairo embeds a
# fallback font for them (the png device is already cairo-based).
ggsave("figures/generated/interp_prefix_reveal_stacked.pdf", out,
       width = CANVAS_W, height = CANVAS_H, device = cairo_pdf)
# 600 dpi: at 300 the thin strokes of the small italic headers anti-alias
# into grey and look faded; the PDF (vector text) is unaffected.
ggsave("figures/generated/interp_prefix_reveal_stacked.png", out,
       width = CANVAS_W, height = CANVAS_H, dpi = 600, bg = "white")
cat("wrote figures/generated/interp_prefix_reveal_stacked.{pdf,png}\n")
