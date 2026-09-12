"""
Runs the forward/backward hierarchical-merge experiment on a folder of
source texts (books or papers). For each text: splits into chunks,
summarizes each chunk, then merges the chunk summaries into a running
summary twice -- once forward (first chunk to last) and once backward
(last chunk to first) -- using GPT-4o mini.

$env:OPENAI_API_KEY = "your_actual_api_key_here"
python merge_order_bias.py --input_dir "text_files/Novels or Papers" --output_dir results --doc_type novel/paper

"""

import os
import re
import json
import argparse
import tiktoken
from openai import OpenAI


MODEL_NAME = "gpt-4o-mini"
MAX_CHUNK_TOKENS = 6000

client = OpenAI()  # reads OPENAI_API_KEY from the environment
encoding = tiktoken.encoding_for_model(MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))
print("OpenAI client ready.")


# Generate text from a prompt
def generate(prompt: str, max_new_tokens: int = 300) -> str:

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return response.choices[0].message.content


# chunking the books or papers to chapters or sections
def split_chunks(full_text: str) -> list[str]:

    if "###CHAPTER###" in full_text:
        # Manual markers
        chunks = full_text.split("###CHAPTER###")
    elif "###SECTION###" in full_text:
        chunks = full_text.split("###SECTION###")
    elif re.search(r"\n\s*[IVXLCDM]+\s*\n", full_text):
        # Roman-numeral chapter headings
        chunks = re.split(r"\n\s*[IVXLCDM]+\s*\n", full_text)
    elif re.search(r"\n\s*(Letter|Chapter)\s+\d+\s*\n", full_text):
        chunks = re.split(r"\n\s*(?:Letter|Chapter)\s+\d+\s*\n", full_text)
    else:
        # Numbered section headings, e.g. "1. Introduction" or "2 Related Work"
        chunks = re.split(r"\n\s*\d+\.?\s+[A-Z][a-zA-Z\s]{2,40}\n", full_text)
    return [c.strip() for c in chunks if c.strip()]


# Split a chapter into smaller pieces if it exceeds max_tokens, splitting on paragraph breaks
def split_long_chunk(chunk_text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:

    if count_tokens(chunk_text) <= max_tokens:
        return [chunk_text]

    """
    paragraphs one at a time, packing them into a "current" bucket. 
    Whenever adding the next paragraph would overflow the token limit, 
    it seals the bucket, starts a new one, and keeps going
    """

    paragraphs = chunk_text.split("\n\n")
    sub_chunks, current = [], ""

    for paragraph in paragraphs:
        candidate = current + "\n\n" + paragraph if current else paragraph
        if count_tokens(candidate) > max_tokens and current:
            sub_chunks.append(current) # save what we've built so far as a finished piece, and start a new piece with just this paragraph
            current = paragraph
        else:
            current = candidate
    if current:
        sub_chunks.append(current)
    return sub_chunks


# Summarize each chunk
def summarize_chunk(chunk_text: str) -> str:

    pieces = split_long_chunk(chunk_text)
    if len(pieces) == 1:
        prompt = f"""Summarize the following text in a concise paragraph, preserving key content, entities, and details:

    {pieces[0]}

    Summary:"""
        return generate(prompt, max_new_tokens=250)


    piece_summaries = []
    for piece in pieces:
        prompt = f"""Summarize the following excerpt in a concise paragraph, preserving key content, entities, and details:

        {piece}

        Summary:"""
        # Summarize each piece, then combine those piece-summaries into one chapter summary
        piece_summaries.append(generate(prompt, max_new_tokens=200))


    combine_prompt = f"""Combine the following partial summaries of one section into a single concise summary:

    {chr(10).join(piece_summaries)}

    Combined summary:"""
    return generate(combine_prompt, max_new_tokens=250)


# Forward and Backward Merge Pipeline sequentially
def merge_prompt(running_summary: str, new_chunk_summary: str, chunks_so_far_count: int, direction: str) -> str:

    if direction == "forward":
        context_note = f"Previous summary (covers sections 1-{chunks_so_far_count - 1}):"
    else:
        context_note = (
            f"Previous summary (covers {chunks_so_far_count - 1} sections "
            "processed so far, from the end of the document backward):"
        )

    reverse_note = "" if direction == "forward" else (
        "\n5. Note: sections are being processed out of their original document order "
        "(starting from the end). Just summarize the content given -- don't try to "
        "reorder it back into original sequence."
    )

    return f"""You are maintaining a running summary of a document, one section at a time. This mirrors how real long-document summarization systems work: at each step, you have a summary of everything so far, and you must produce a new summary that reads naturally as ONE continuous piece of text -- not a list.

    {context_note}
    {running_summary}

    New section content to add:
    {new_chunk_summary}

    Your task:
    1. Write a new summary covering all {chunks_so_far_count} sections processed so far, as flowing prose.
    2. You MUST weave in specific new details from the new section just given -- do not skip or gloss over it.
    3. You are free to compress, shorten, or drop minor detail from previously-processed sections as needed to fit the new content in -- you do not need to preserve every earlier detail verbatim.
    4. Do not simply copy the previous summary and append a sentence -- genuinely rewrite it as one coherent piece.{reverse_note}

    Updated summary:"""


# Run hierarchical merging over chapter_summaries
def run_merge(chunk_summaries: list[str], direction: str) -> tuple[str, list[dict]]:

    """
    merge_log always records ORIGINAL chapter numbers,
    regardless of the order they were processed
    """
    assert direction in ("forward", "backward")
    order = list(range(len(chunk_summaries))) if direction == "forward" else list(range(len(chunk_summaries) - 1, -1, -1))

    merge_log = []
    first_idx = order[0]
    running_summary = chunk_summaries[first_idx]
    merge_log.append({"step": 0, "chunks_included": [first_idx + 1], "summary": running_summary})
    chunks_processed = [first_idx + 1]

    for step, idx in enumerate(order[1:], start=1):
        chunks_so_far_count = step + 1
        prompt = merge_prompt(running_summary, chunk_summaries[idx], chunks_so_far_count, direction)
        running_summary = generate(prompt, max_new_tokens=2000)
        chunks_processed.append(idx + 1)
        merge_log.append({"step": step, "chunks_included": sorted(chunks_processed), "summary": running_summary})
        print(f"  [{direction}] step {step}: merged section {idx + 1}")

    return running_summary, merge_log


def truncate(text: str, max_length: int = 20) -> str:
    """Shorten a string to max_length characters, without cutting mid-word if possible."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length].rsplit(" ", 1)[0]
    return truncated if truncated else text[:max_length]

# Main batch loop: process every book or paper, both directions
def process_file(filepath: str, output_dir: str, doc_type: str) -> None:

    filename = os.path.basename(filepath)
    file_id = truncate(filename.replace(".txt", ""))
    title = file_id.replace("_", " ").title()

    forward_path = os.path.join(output_dir, f"{file_id}_forward.json")
    backward_path = os.path.join(output_dir, f"{file_id}_backward.json")

    # Skip if both results already exist
    if os.path.exists(forward_path) and os.path.exists(backward_path):
        print(f"\n=== Skipping: {title} (results already exist) ===")
        return

    print(f"\n=== Processing: {title} ===")
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_chunks(full_text)
    print(f"  Found {len(chunks)} sections.")

    # Pre-check: show how many pieces each chapter will become before spending API calls
    for i, ch in enumerate(chunks):
        pieces = split_long_chunk(ch)
        if len(pieces) > 1:
            print(f"    Section {i + 1} will be split into {len(pieces)} sub-chunks")
        else:
            print(f"    Chapter {i+1}: OK, no split needed")

    print("  Summarizing sections...")
    chunk_summaries = [summarize_chunk(c) for c in chunks]

    print("  Running forward merge...")
    final_forward, log_forward = run_merge(chunk_summaries, "forward")

    print("  Running backward merge...")
    final_backward, log_backward = run_merge(chunk_summaries, "backward")

    # Save results
    for direction, final_summary, merge_log in [
        ("forward", final_forward, log_forward),
        ("backward", final_backward, log_backward),
    ]:
        results = {
            "title": title,
            "type": doc_type,
            "model": MODEL_NAME,
            "merge_order": direction,
            "num_chunks": len(chunks),
            "chunk_summaries": chunk_summaries,
            "merge_log": merge_log,
            "final_summary": final_summary,
        }
        out_path = os.path.join(output_dir, f"{file_id}_{direction}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")


def main():

    parser = argparse.ArgumentParser(description="Run forward/backward hierarchical-merge experiment.")
    parser.add_argument("--input_dir", required=True, help="Folder of .txt source files (books or papers).")
    parser.add_argument("--output_dir", required=True, help="Folder to write result JSON files to.")
    parser.add_argument("--doc_type", required=True, help="Type of file, novel or paper")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]
    print(f"Found {len(input_files)} input files: {input_files}")

    for filename in input_files:
        process_file(os.path.join(args.input_dir, filename), args.output_dir, args.doc_type)

    print("\n=== Batch complete ===")


if __name__ == "__main__":
    main()

