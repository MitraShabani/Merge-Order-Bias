Merge-Order Bias in Book Summarization

Research Question

FABLES (Kim et al., 2024) found that LLM-generated book summaries systematically over-represent content near the end of the book. 
This project tests whether that effect is a genuine model limitation, or an artifact of forward-sequential hierarchical merging — the standard method (used by BooookScore/FABLES) where a running summary is built by repeatedly merging in the next chapter, 
always placing the newest content at the "freshest" point in the process.

Method
1.Split book into chapters (using the book's own structural markers)
2.Summarize each chapter individually
3.Merge chapter summaries sequentially — forward and backward — using GPT-4o mini
4.Compare per-chapter coverage/detail between the two directions
