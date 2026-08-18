# Merge-Order Bias in Book Summarization
## Question

FABLES (Kim et al., 2024) found LLM book summaries over-emphasize content near the end of the book, using forward-only merging (chapter 1→2→...→last). This project tests whether that's a real model limitation, or an artifact of processing chapters in that fixed order.

## Method

For each book: split into chapters, summarize each individually, merge them into a running summary two ways, forward and backward, using GPT-4o mini.

Two metrics were used:

1. Stability: for each chapter, embedding similarity between the running summary right when that chapter was added vs. the final summary. Measures how much a chapter's surrounding text keeps getting rewritten after it joins, not whether its specific content survives.

2. Coverage: for each chapter, extract 4 specific facts from its original summary, then check whether each fact is still present in the final merged summary. Coverage score = fraction of facts that survived. This is the metric that directly answers the research question (stability measures rewrite frequency; coverage measures actual content survival).

## Findings 
(3 books: The Gambler, Dr Jekyll and Mr Hyde, Pride and Prejudice)

Stability: forward merging is consistently more stable than backward merging, higher mean, lower variance, across all 3 books.

| Book |	Forward Mean ± Std	| Backward Mean ± Std |
|------|--------------------|---------------------|
| Pride and Prejudice |	0.915 ± 0.043	| 0.843 ± 0.032 |
| Dr Jekyll and Mr Hyde |	0.977 ± 0.036	| 0.761 ± 0.082 |
| The Gambler	| 0.991 ± 0.022	| 0.762 ± 0.058 |

Coverage: forward merging also preserves more chapter content than backward merging, consistently across all 3 books:

| Book	| Forward Coverage	| Backward Coverage |
|------|------------------|-----------------|
| Dr Jekyll and Mr Hyde	| 0.977	| 0.364 |
| The Gambler	| 0.639	| 0.597 |
| Pride and Prejudice	| 0.337	| 0.219 |

The size of the coverage gap varies a lot between books and doesn't correlate cleanly with chapter count, not a uniform effect, but the direction (forward > backward) held every time, on both metrics.

Separately: longer books show lower overall coverage in both directions (chapter count vs. forward coverage: r = -0.93, n=3), content preservation seems to degrade with book length, independent of merge direction. Worth noting, not yet confirmed with more books.

## What this does and doesn't show

Supports: merge order affects both how consistent the merge process is (stability) and how well chapter content survives (coverage) — a real, measurable, replicated effect across two independent metrics.

Does not cleanly support: a "mirror image" pattern where the effect flips symmetrically between the book's beginning and end depending on direction. Only one of three books (The Gambler) showed that shape in coverage; the other two showed forward simply better overall, not a clean positional flip.


## Background
Chang et al., "BooookScore" (ICLR 2024)
Kim et al., "FABLES" (2024)
