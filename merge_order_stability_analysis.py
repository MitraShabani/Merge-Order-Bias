import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


# A book is "complete" if both its forward and backward files exist
def find_result_pairs(results_dir: str) -> list[str]:
    all_files = os.listdir(results_dir)
    files_id = set(f.replace("_forward.json", "").replace("_backward.json", "")
                for f in all_files if f.endswith(".json"))
    pairs = []
    for id in files_id:
        if f"{id}_forward.json" in all_files and f"{id}_backward.json" in all_files:
            pairs.append(id)
        else:
            print(f"WARNING: {id} is missing forward or backward results -- skipping")
    return sorted(pairs)


# Extract each chapter's FIRST and LAST appearance text
def get_first_appearance_text(merge_log: list[dict], chunk_num: int) -> str | None:

    """ Return the full summary text from the step where `chapter_num` was first introduced."""

    key = "chapters_included" if "chapters_included" in merge_log[0] else "chunks_included"
    for step in merge_log:
        if chunk_num in step[key]:
            return step["summary"]
    return None


def get_last_appearance_text(merge_log: list[dict], chunk_num: int) -> str | None:

    """ Return the full summary text from the LAST step that includes `chapter_num`
    (in this design, that's simply the final step, since chapters are never dropped)."""

    key = "chapters_included" if "chapters_included" in merge_log[0] else "chunks_included"
    for step in reversed(merge_log):
        if chunk_num in step[key]:
            return step["summary"]
    return None

# Compute stability: Compute stability per chapter, per direction
def compute_stability(merge_log: list[dict], num_chunks: int, embedder) -> list[dict]:

    """ For each chapter: compare its original standalone summary to (a) the step where it was first introduced, and (b) the FINAL summary as a whole.
    The gap between (a) and (b) tells us how much that chapter's presence/similarity degraded from its first appearance to the end of the merge process."""

    results = []
    for i in range(num_chunks):
        chapter_num = i + 1
        first_text = get_first_appearance_text(merge_log, chapter_num)
        last_text = get_last_appearance_text(merge_log, chapter_num)

        emb_first = embedder.encode(first_text, convert_to_tensor=True)
        emb_last = embedder.encode(last_text, convert_to_tensor=True)

        stability = util.cos_sim(emb_first, emb_last).item()
        results.append({"chapter": chapter_num, "stability": stability})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run stability analysis on merge results.")
    parser.add_argument("--results_dir", required=True, help="Folder with *_forward.json / *_backward.json files.")
    parser.add_argument("--plots_dir", required=True, help="Folder to write plots and tables to.")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    pairs = find_result_pairs(args.results_dir)
    print(f"Found {len(pairs)} complete pairs: {pairs}")

    summary_data = []

    for doc_id in pairs:
        with open(os.path.join(args.results_dir, f"{doc_id}_forward.json"), "r", encoding="utf-8") as f:
            forward = json.load(f)
        with open(os.path.join(args.results_dir, f"{doc_id}_backward.json"), "r", encoding="utf-8") as f:
            backward = json.load(f)

        title = forward.get("title") or forward.get("book_title", doc_id)
        num_chapters = forward.get("num_chunks") or forward.get("num_chapters")
        doc_type = forward.get("type", "unknown")
        print(f"\n=== {title} ({num_chapters} chunks, type={doc_type}) ===")

        forward_stability = compute_stability(forward["merge_log"], num_chapters, embedder)
        backward_stability = compute_stability(backward["merge_log"], num_chapters, embedder)

        # Plot: stability (first-appearance vs. final) by chapter position,, forward vs. backward
        chapters = [d["chapter"] for d in forward_stability]
        f_vals = [d["stability"] for d in forward_stability]
        b_vals = [d["stability"] for d in backward_stability]

        plt.figure(figsize=(10, 6))
        plt.plot(chapters, f_vals, marker="o", label="Forward merge", color="steelblue")
        plt.plot(chapters, b_vals, marker="s", label="Backward merge", color="firebrick")
        plt.xlabel("chapter number (document order)")
        plt.ylabel("Stability: first-appearance vs. final-step similarity")
        plt.title(f"{title}: stability by chapter, forward vs. backward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(chapters)
        plot_path = os.path.join(args.plots_dir, f"stability_{doc_type}_{doc_id}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path}")


        # For cross-book summary
        # Summary stats using ALL chapters, not just the first-processed one
        summary_data.append({
            "document": title,
            "type": doc_type,
            "num_chapters": num_chapters,
            "forward_mean": float(np.mean(f_vals)),
            "forward_std": float(np.std(f_vals)),
            "backward_mean": float(np.mean(b_vals)),
            "backward_std": float(np.std(b_vals)),
        })
        print(f"  Forward:  mean={summary_data[-1]['forward_mean']:.3f}, std={summary_data[-1]['forward_std']:.3f}")
        print(f"  Backward: mean={summary_data[-1]['backward_mean']:.3f}, std={summary_data[-1]['backward_std']:.3f}")


    # Cross-book summary table
    # Mean and standard deviation of stability across ALL chapters

    """ Mean shows the typical stability level;
    standard deviation shows how consistent that level is across chapters."""

    summary_df = pd.DataFrame(summary_data)
    summary_table_path = os.path.join(args.plots_dir, "stability_summary_table.csv")
    summary_df.to_csv(summary_table_path, index=False)

    print(f"\nSaved summary table: {summary_table_path}")
    print(summary_df.round(3).to_string(index=False))

    # Cross-document summary plot, grouped by type
    for doc_type, group in summary_df.groupby("type"):
        docs = group["document"].tolist()

        x = np.arange(len(docs))
        width = 0.35

        plt.figure(figsize=(9, 6))
        plt.bar(x - width / 2, group["forward_mean"], width, yerr=group["forward_std"], capsize=5,
                label="Forward", color="steelblue")
        plt.bar(x + width / 2, group["backward_mean"], width, yerr=group["backward_std"], capsize=5,
                label="Backward", color="firebrick")
        plt.xlabel("Document")
        plt.ylabel("Mean stability (error bars = std across chunks)")
        plt.title(f"Mean stability, forward vs. backward -- type: {doc_type}")
        plt.xticks(x, docs, rotation=15, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        summary_plot_path = os.path.join(args.plots_dir, f"stability_summary_{doc_type}.png")
        plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved summary plot ({doc_type}): {summary_plot_path}")

    print("\n=== Stability analysis complete ===")


if __name__ == "__main__":
    main()
