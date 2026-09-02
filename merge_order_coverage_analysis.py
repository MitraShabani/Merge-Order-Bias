import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI


MODEL_NAME = "gpt-4o-mini"
N_FACTS_PER_CHUNK = 4
client = OpenAI()  # reads OPENAI_API_KEY from the environment


def generate(prompt: str, max_new_tokens: int = 300) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return response.choices[0].message.content


def find_result_pairs(results_dir: str) -> list[str]:

    all_files = os.listdir(results_dir)
    files_id = set(f.replace("_forward.json", "").replace("_backward.json", "")
                for f in all_files if f.endswith(".json") and "_coverage" not in f)
    pairs = []
    for doc_id in files_id:
        if f"{doc_id}_forward.json" in all_files and f"{doc_id}_backward.json" in all_files:
            pairs.append(doc_id)
        else:
            print(f"WARNING: {doc_id} is missing forward or backward results -- skipping")
    return sorted(pairs)


# Fact extraction and coverage check
def extract_facts(chunk_summary: str, n_facts: int = N_FACTS_PER_CHUNK) -> list[str]:

    prompt = f"""Extract exactly {n_facts} short, specific, checkable facts from the following summary. Each fact should be one short sentence, checkable as true/false, and specific enough that a paraphrase of it would still count (e.g., a named event, an action, a specific outcome). Do not include vague or generic facts.

Summary:
{chunk_summary}

Output ONLY the {n_facts} facts, one per line, no numbering, no extra text."""
    result = generate(prompt, max_new_tokens=200)
    facts = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return facts[:n_facts]



def fact_present(fact: str, final_summary: str) -> bool:
    prompt = f"""Does the following summary mention or clearly imply this fact (even if paraphrased differently)? Answer with ONLY "yes" or "no".

Fact: {fact}

Summary:
{final_summary}

Answer (yes/no):"""
    result = generate(prompt, max_new_tokens=5).strip().lower()
    return result.startswith("yes")



def compute_coverage(chunk_summaries: list[str], final_summary: str, n_facts: int = N_FACTS_PER_CHUNK) -> list[dict]:

    """For each chapter, extract facts and check how many survive into final_summary.
    Returns a list of dicts: {chapter, facts, coverage}."""

    results = []
    for i, chunk_summary in enumerate(chunk_summaries):
        chunk_num = i + 1
        facts = extract_facts(chunk_summary, n_facts=n_facts)
        found = [fact_present(f, final_summary) for f in facts]
        coverage = sum(found) / len(facts) if facts else 0
        results.append({"chapter": chunk_num, "facts": facts, "found": found, "coverage": coverage})
        print(f"  Chapter {chunk_num}: {sum(found)}/{len(facts)} facts found (coverage={coverage:.2f})")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run coverage analysis on merge results.")
    parser.add_argument("--results_dir", required=True, help="Folder with *_forward.json / *_backward.json files.")
    parser.add_argument("--plots_dir", required=True, help="Folder to write plots and tables to.")
    parser.add_argument("--n_facts", type=int, default=N_FACTS_PER_CHUNK, help="Facts to extract per chunk.")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    pairs = find_result_pairs(args.results_dir)
    print(f"Found {len(pairs)} complete pairs: {pairs}")

    # Coverage analysis for each book
    coverage_summary = []

    for doc_id in pairs:
        with open(os.path.join(args.results_dir, f"{doc_id}_forward.json"), "r", encoding="utf-8") as f:
            forward = json.load(f)
        with open(os.path.join(args.results_dir, f"{doc_id}_backward.json"), "r", encoding="utf-8") as f:
            backward = json.load(f)

        title = forward.get("title") or forward.get("book_title", doc_id)
        chunk_summaries = forward.get("chunk_summaries") or forward.get("chapter_summaries")
        doc_type = forward.get("type", "unknown")
        print(f"\n=== {title} (type={doc_type}) ===")

        print(" Forward coverage:")
        forward_coverage = compute_coverage(chunk_summaries, forward["final_summary"], args.n_facts)
        print(" Backward coverage:")
        backward_coverage = compute_coverage(chunk_summaries, backward["final_summary"], args.n_facts)

        # Save raw coverage data
        coverage_data = {"title": title, "type": doc_type,
                          "forward_coverage": forward_coverage, "backward_coverage": backward_coverage}
        out_path = os.path.join(args.results_dir, f"{doc_id}_coverage.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coverage_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

        chapters = [d["chapter"] for d in forward_coverage]
        f_cov = [d["coverage"] for d in forward_coverage]
        b_cov = [d["coverage"] for d in backward_coverage]

        plt.figure(figsize=(10, 6))
        plt.plot(chapters, f_cov, marker="o", label="Forward merge", color="steelblue")
        plt.plot(chapters, b_cov, marker="s", label="Backward merge", color="firebrick")
        plt.xlabel("chapter number (document order)")
        plt.ylabel("Fact coverage (fraction of chapter's facts found in final summary)")
        plt.title(f"{title}: fact coverage by chapter, forward vs. backward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(chapters)
        plt.ylim(-0.05, 1.05)
        plot_path = os.path.join(args.plots_dir, f"coverage_{doc_type}_{doc_id }.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path}")

        coverage_summary.append({
            "document": title,
            "type": doc_type,
            "forward_mean_coverage": float(np.mean(f_cov)),
            "backward_mean_coverage": float(np.mean(b_cov)),
        })

    # Cross-book summary table
    coverage_df = pd.DataFrame(coverage_summary)
    table_path = os.path.join(args.plots_dir, "coverage_summary_table.csv")
    coverage_df.to_csv(table_path, index=False)

    print(f"\nSaved summary table: {table_path}")
    print(coverage_df.round(3).to_string(index=False))

    print("\n=== Coverage analysis complete ===")


if __name__ == "__main__":
    main()
