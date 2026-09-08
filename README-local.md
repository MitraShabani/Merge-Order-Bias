# Merge-Order Bias in Hierarchical Summarization
## Question

FABLES (Kim et al., 2024) found LLM book summaries over-emphasize content near the end of the book, using forward-only merging (chapter 1→2→...→last). This project tests whether that effect is a real model limitation, or an artifact of processing chunks in that fixed order, and whether it holds the same way in narrative (novels) vs. non-narrative (scientific papers) text.

## Method

For each document: split into chapters, summarize each individually, merge them into a running summary two ways, forward and backward, using GPT-4o mini.

Two metrics were used:

1. Stability: for each chapter, embedding similarity between a chunk's text right when it was first merged in vs. the final summary. Measures how much a chunk's surrounding text keeps changing after it joins.

2. Coverage: for each chunk, extract 4 specific facts, then check whether each survives into the final summary. Measures actual content preservation, not just wording change

## Findings
(8 documents: 4 novels, The Gambler, Dr Jekyll and Mr Hyde,Pride and Prejudice, Frankenstein. 4 papers, Bean et al. 2025, Rudinger et al. 2018, Łajewska et al. 2025, Qin et al. 2025.)

Forward merging shows a consistent advantage over backward in scientific papers, but not reliably in novels:

| Text type | Stability: Forward > Backward	| Coverage: Forward > Backward |
|--------|--------------------------------------|------------------------------|
| Papers (4 documents) |	4/4	| 4/4 |
| Novels (4 documents) |	2/4	| 1/4 |


In papers, forward merging is consistently more stable and higher-coverage than backward, with no exceptions across all 4 documents tested. In novels, the pattern is inconsistent, half or more of the books show backward equal to or higher than forward on at least one metric.

## What this does and doesn't show

This does not cleanly answer the original question either way. If merge order were purely a procedural artifact independent of content, we would expect the effect to appear equally in both text type. If it were purely about narrative content (e.g., "endings" specifically), we would expect it to appear in novels, not papers. Instead, the effect is strong and consistent in the domain without narrative structure, and weak/inconsistent in the domain with it, the opposite of what a simple "it's just about endings" story would predict, but not a clean confirmation of "it's purely mechanical" either.

This is treated as a genuine, open finding: merge order has a real, measurable effect, but its reliability depends on text type in a way that isn't yet explained.

## Limitations
Small sample (4 documents per text type), the novel/paper split is a real, consistent pattern in this data, but not large enough to rule out document-specific factors within each set
Single model (GPT-4o mini), whether this generalizes to other LLMs is untested
GPT-4o mini is not perfectly deterministic even at temperature=0; results regenerated from scratch showed some run-to-run variation on the novel coverage numbers specifically

## Background
- Chang et al., "BooookScore" (ICLR 2024)
- Kim et al., "FABLES" (2024)
- Olabisi & Agrawal, "Understanding Position Bias Effects on Fairness in Social Multi-Document Summarization" (2024)
- "On Positional Bias of Faithfulness for Long-form Summarization" (NAACL 2025) — tests position bias on ArXiv/PubMed but not full-reversal merge order specifically
