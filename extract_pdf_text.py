import os
import argparse
from pypdf import PdfReader


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs into .txt files.")
    parser.add_argument("--input_dir", required=True, help="Folder of .pdf files.")
    parser.add_argument("--output_dir", required=True, help="Folder to write .txt files to.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    for filename in pdf_files:
        pdf_path = os.path.join(args.input_dir, filename)
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(args.output_dir, txt_filename)

        if os.path.exists(txt_path):
            print(f"  Skipping {filename} (already extracted)")
            continue

        print(f"  Extracting: {filename}")
        text = extract_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"    Saved: {txt_path} ({len(text)} characters)")

    print("\n=== Extraction complete ===")
    print("NOTE: check each .txt file for section headings before running the merge pipeline.")
    print("If sections aren't detected automatically, insert '###SECTION###' markers manually.")


if __name__ == "__main__":
    main()
