# Merge-Order Bias in Book Summarization
## Question

FABLES (Kim et al., 2024) found LLM book summaries over-emphasize content near the end of the book, using forward-only merging (chapter 1→2→...→last). This project tests whether that's a real model limitation, or an artifact of processing chapters in that fixed order.

## Method

For each book: split into chapters, summarize each individually, merge them into a running summary two ways, forward and backward, using GPT-4o mini.

## Done so far

Built a stability metric: for each chapter, compare the running summary right when that chapter was added vs. the final summary (embedding cosine similarity). Tested on 3 books (The Gambler, Dr Jekyll and Mr Hyde, Pride and Prejudice).

Result: forward merging is consistently more stable than backward merging across all 3 books. But on inspection, this metric doesn't answer the original question — it measures how much the whole running summary changes after a chapter joins (which mechanically favors whichever chapter is processed last), not whether that chapter's actual content survived. Stability ≠ coverage.
|  | Books | Chapters | Forward Mean | Forward std | Backward Mean | Backward std |
|--|-------|----------|--------------|-------------|---------------|--------------|
|0| Pride and Prejudice| 49 | 0.915 | 0.043 | 0.843 | 0.032 |
|1| The Strange Case Of Dr.Jekyll and Mr.Hyde | 11 | 0.977 | 0.036 | 0.761 |0.082 |
|2| Gambler | 18 | 0.991 | 0.022 | 0.762 | 0.058 |

## Next: coverage metric

To directly test the original question, need to measure fact survival, not wording stability:

Extract 3-5 key facts per chapter (from its original summary)
Check whether each fact appears in the final summary
Coverage score per chapter = fraction of facts that survived
Compare coverage by chapter position, forward vs. backward

This is the actual test: if coverage shifts toward whichever end is processed last (flips between forward/backward) → artifact of merge order. If coverage stays fixed on the book's real ending regardless of direction → real model limitation.

## Status
 Forward/backward merge pipeline, batch-processes any book in books/
 Stability metric built and tested on 3 books (see limitation above)
 Coverage metric (fact-extraction based) — not yet built
 Answer to the core research question — still open

## Background
Chang et al., "BooookScore" (ICLR 2024)
Kim et al., "FABLES" (2024)
