"""
Download and process WMDP-Bio-Retain corpus from Hugging Face
"""

import json
import random
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset


DEFAULT_TARGET_SIZE = 5300
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = 'dataset'
DEFAULT_FORGET_RATIOS = [0.01, 0.05, 0.10, 0.20]


def download_wmdp_bio_retain():
    print("Downloading WMDP-Bio-Retain corpus from Hugging Face...")
    try:
        dataset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus")
        print(f"✓ Dataset downloaded successfully")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure you have the 'datasets' library installed:")
        print("  pip install datasets")
        return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    invisible_chars = ['\u200b','\u200c','\u200d','\u200e','\u200f','\ufeff','\u202a','\u202b','\u202c','\u202d','\u202e']
    for char in invisible_chars:
        text = text.replace(char, '')
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\v', ' ').replace('\f', ' ').replace('\x85', ' ').replace('\u2028', ' ').replace('\u2029', ' ')
    lines = text.split('  ')
    cleaned_lines = []
    for line in lines:
        if line.count('\t') >= 2 or line.count('|') >= 3:
            continue
        if re.search(r'[-=_]{4,}', line):
            continue
        cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)
    text = text.replace('\t', ' ')
    text = re.sub(r'\[?\d+\]', '', text)
    text = re.sub(r'et al\.?\s*,?', 'et al.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = text.replace('\u00ad', '')
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    return text


def is_valid_sentence(text: str, min_length: int = 30, max_length: int = 1000) -> bool:
    if not text:
        return False
    if len(text) < min_length or len(text) > max_length:
        return False
    if not re.search(r'[a-zA-Z]', text):
        return False
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?()\-\'\"]', text)) / len(text)
    if special_char_ratio > 0.15:
        return False
    digit_ratio = len(re.findall(r'\d', text)) / len(text)
    if digit_ratio > 0.3:
        return False
    if not re.search(r'[.!?]$', text.strip()):
        return False
    return True


def extract_sentences_from_dataset(dataset, target_size: int, seed: int = 42) -> List[Dict]:
    print(f"\nExtracting sentences from PubMed papers...")
    random.seed(seed)
    sentences = []
    filtered_count = 0
    split_data = dataset['train']
    print(f"Processing train split ({len(split_data)} papers)...")
    for idx, paper in enumerate(split_data):
        text_content = paper.get('text', '').strip()
        if text_content:
            cleaned_text = clean_text(text_content)
            paper_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
            for sent_idx, sentence in enumerate(paper_sentences):
                cleaned_sentence = clean_text(sentence)
                if is_valid_sentence(cleaned_sentence):
                    sentence_data = {'text': cleaned_sentence, 'source': 'wmdp-bio-retain', 'type': 'paper_sentence', 'paper_index': idx, 'sentence_index': sent_idx}
                    sentences.append(sentence_data)
                else:
                    filtered_count += 1
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} papers, extracted {len(sentences)} sentences so far...")
    print(f"\n✓ Extracted {len(sentences)} total sentences from PubMed papers")
    print(f"  Filtered out {filtered_count} low-quality sentences")
    if len(sentences) > target_size:
        print(f"Sampling {target_size} sentences from {len(sentences)} available...")
        random.shuffle(sentences)
        sentences = sentences[:target_size]
    elif len(sentences) < target_size:
        print(f"Warning: Only found {len(sentences)} sentences (target: {target_size})")
        print(f"Using all {len(sentences)} available sentences")
    print(f"\n✓ Final dataset size: {len(sentences)} sentences")
    return sentences


def save_jsonl(data: List[Dict], filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Dict], filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def create_forget_retain_splits(sentences: List[Dict], forget_ratios: List[float], seed: int, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    total_sentences = len(sentences)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Total sentences: {total_sentences}")
    random.seed(seed)
    shuffled_data = sentences.copy()
    random.shuffle(shuffled_data)
    full_file = output_path / 'wmdp_bio_full.json'
    print(f"\nSaving {full_file}...")
    save_json(shuffled_data, full_file)
    full_jsonl = output_path / 'wmdp_bio_full.jsonl'
    print(f"Saving {full_jsonl}...")
    save_jsonl(shuffled_data, full_jsonl)
    for ratio in forget_ratios:
        forget_size = int(total_sentences * ratio)
        forget_name = str(int(ratio * 100)).zfill(2)
        print(f"\n--- Creating forget{forget_name} split ({ratio*100}% = {forget_size} sentences) ---")
        forget_data = shuffled_data[:forget_size]
        retain_data = shuffled_data[forget_size:]
        forget_file = output_path / f'wmdp_bio_forget{forget_name}.json'
        print(f"Saving {forget_file} ({len(forget_data)} sentences)...")
        save_json(forget_data, forget_file)
        retain_file = output_path / f'wmdp_bio_retain{forget_name}.json'
        print(f"Saving {retain_file} ({len(retain_data)} sentences)...")
        save_json(retain_data, retain_file)
        forget_jsonl = output_path / f'wmdp_bio_forget{forget_name}.jsonl'
        save_jsonl(forget_data, forget_jsonl)
        retain_jsonl = output_path / f'wmdp_bio_retain{forget_name}.jsonl'
        save_jsonl(retain_data, retain_jsonl)
    print("\n✓ Data splitting complete!")
    print(f"\nAll files saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Download and process WMDP-Bio-Retain corpus for unlearning experiments")
    parser.add_argument('--target_size', type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--forget_ratios', type=float, nargs='+', default=DEFAULT_FORGET_RATIOS)
    args = parser.parse_args()
    print("WMDP-BIO-RETAIN DATASET PREPARATION")
    dataset = download_wmdp_bio_retain()
    if dataset is None:
        return
    sentences = extract_sentences_from_dataset(dataset=dataset, target_size=args.target_size, seed=args.seed)
    if not sentences:
        print("Error: No sentences extracted from dataset")
        return
    create_forget_retain_splits(sentences=sentences, forget_ratios=args.forget_ratios, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
