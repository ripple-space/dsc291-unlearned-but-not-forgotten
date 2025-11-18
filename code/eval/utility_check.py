"""
Utility Check Script

Compares model utility (accuracy/perplexity) on retain set between:
- Oracle model (trained on full dataset)
- Unlearned model (trained on full_minus_forget dataset)

Generates scatter plots and line comparisons to visualize utility preservation.

Based on the paper "Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs"

Usage:
    python utility_check.py \
        --oracle_model models/llama-3.1-8b-medical-oracle \
        --unlearned_model models/llama-3.1-8b-medical-unlearned \
        --retain_dataset dataset/med_synthetic_full_minus_forget01.json \
        --output_dir results/utility_check \
        --max_samples 200
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from datetime import datetime
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load dataset from JSON or JSONL file."""
    logger.info(f"Loading dataset from: {dataset_path}")

    records = []
    if dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    if max_samples and len(records) >= max_samples:
                        break
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            records = data[:max_samples] if max_samples else data

    logger.info(f"Loaded {len(records)} records")
    return records


def format_medical_soap(record: Dict) -> str:
    """Format medical SOAP note for evaluation."""
    client_name = record.get('client_name', 'Unknown')
    dob = record.get('date_of_birth', 'Unknown')
    date = record.get('date', 'Unknown')
    subjective = record.get('subjective', '')
    objective = record.get('objective', '')
    assessment = record.get('assessment', '')
    plan = record.get('plan', '')

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical documentation assistant. Given patient information, create a comprehensive SOAP note (Subjective, Objective, Assessment, Plan).<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a SOAP note for the following patient:
- Patient Name: {client_name}
- Date of Birth: {dob}
- Visit Date: {date}

Patient Presentation:
{subjective}

Physical Examination Findings:
{objective}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

SOAP Note for {client_name}

Date of Birth: {dob}
Visit Date: {date}

Subjective:
{subjective}

Objective:
{objective}

Assessment:
{assessment}

Plan:
{plan}<|eot_id|>"""


def format_wmdp(record: Dict) -> str:
    """Format WMDP record for evaluation."""
    return record.get('text', '')


def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute perplexity for a given text.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text

    Returns:
        Perplexity score
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings['input_ids'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def compute_next_token_accuracy(model, tokenizer, text: str, num_predictions: int = 50) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        num_predictions: Number of next-token predictions to evaluate

    Returns:
        Accuracy score (0-1)
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings['input_ids'].to(model.device)

    if input_ids.shape[1] < num_predictions + 1:
        num_predictions = input_ids.shape[1] - 1

    if num_predictions <= 0:
        return 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(num_predictions):
            context = input_ids[:, :-(num_predictions - i)]
            target = input_ids[:, -(num_predictions - i)]

            outputs = model(context)
            logits = outputs.logits
            predicted = torch.argmax(logits[:, -1, :], dim=-1)

            if predicted.item() == target.item():
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_model_utility(
    model,
    tokenizer,
    records: List[Dict],
    dataset_type: str
) -> Dict[str, List[float]]:
    """
    Evaluate model utility on retain set.

    Args:
        model: Language model
        tokenizer: Tokenizer
        records: Dataset records
        dataset_type: 'medical_soap' or 'wmdp'

    Returns:
        Dictionary with perplexity and accuracy lists
    """
    perplexities = []
    accuracies = []

    for record in tqdm(records, desc="Computing utility metrics"):
        # Format text based on dataset type
        if dataset_type == 'medical_soap':
            text = format_medical_soap(record)
        else:
            text = format_wmdp(record)

        # Compute perplexity
        try:
            ppl = compute_perplexity(model, tokenizer, text)
            if not math.isnan(ppl) and not math.isinf(ppl):
                perplexities.append(ppl)
            else:
                perplexities.append(float('inf'))
        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            perplexities.append(float('inf'))

        # Compute next-token accuracy
        try:
            acc = compute_next_token_accuracy(model, tokenizer, text, num_predictions=50)
            accuracies.append(acc)
        except Exception as e:
            logger.warning(f"Failed to compute accuracy: {e}")
            accuracies.append(0.0)

    return {
        'perplexities': perplexities,
        'accuracies': accuracies
    }


def plot_scatter_comparison(
    oracle_metrics: Dict[str, List[float]],
    unlearned_metrics: Dict[str, List[float]],
    output_path: str
):
    """
    Create scatter plot comparing oracle vs unlearned model metrics.

    Args:
        oracle_metrics: Metrics from oracle model
        unlearned_metrics: Metrics from unlearned model
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Perplexity scatter plot
    oracle_ppl = np.array(oracle_metrics['perplexities'])
    unlearned_ppl = np.array(unlearned_metrics['perplexities'])

    # Filter out infinite values for plotting
    valid_idx = np.isfinite(oracle_ppl) & np.isfinite(unlearned_ppl)
    oracle_ppl_valid = oracle_ppl[valid_idx]
    unlearned_ppl_valid = unlearned_ppl[valid_idx]

    ax1.scatter(oracle_ppl_valid, unlearned_ppl_valid, alpha=0.6, s=50)
    ax1.plot([oracle_ppl_valid.min(), oracle_ppl_valid.max()],
             [oracle_ppl_valid.min(), oracle_ppl_valid.max()],
             'r--', linewidth=2, label='y=x (perfect match)')
    ax1.set_xlabel('Oracle Model Perplexity', fontsize=14)
    ax1.set_ylabel('Unlearned Model Perplexity', fontsize=14)
    ax1.set_title('Perplexity Comparison on Retain Set', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy scatter plot
    oracle_acc = np.array(oracle_metrics['accuracies'])
    unlearned_acc = np.array(unlearned_metrics['accuracies'])

    ax2.scatter(oracle_acc, unlearned_acc, alpha=0.6, s=50, color='green')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (perfect match)')
    ax2.set_xlabel('Oracle Model Accuracy', fontsize=14)
    ax2.set_ylabel('Unlearned Model Accuracy', fontsize=14)
    ax2.set_title('Next-Token Accuracy Comparison on Retain Set', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Scatter plot saved to: {output_path}")


def plot_line_comparison(
    oracle_metrics: Dict[str, List[float]],
    unlearned_metrics: Dict[str, List[float]],
    output_path: str
):
    """
    Create line plot showing metric trends across samples.

    Args:
        oracle_metrics: Metrics from oracle model
        unlearned_metrics: Metrics from unlearned model
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Perplexity line plot
    oracle_ppl = np.array(oracle_metrics['perplexities'])
    unlearned_ppl = np.array(unlearned_metrics['perplexities'])

    # Filter out infinite values
    oracle_ppl_finite = np.where(np.isfinite(oracle_ppl), oracle_ppl, np.nan)
    unlearned_ppl_finite = np.where(np.isfinite(unlearned_ppl), unlearned_ppl, np.nan)

    x = np.arange(len(oracle_ppl))
    ax1.plot(x, oracle_ppl_finite, label='Oracle Model', linewidth=2, alpha=0.7)
    ax1.plot(x, unlearned_ppl_finite, label='Unlearned Model', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Sample Index', fontsize=14)
    ax1.set_ylabel('Perplexity', fontsize=14)
    ax1.set_title('Perplexity Across Samples', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Accuracy line plot
    oracle_acc = np.array(oracle_metrics['accuracies'])
    unlearned_acc = np.array(unlearned_metrics['accuracies'])

    ax2.plot(x, oracle_acc, label='Oracle Model', linewidth=2, alpha=0.7)
    ax2.plot(x, unlearned_acc, label='Unlearned Model', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Sample Index', fontsize=14)
    ax2.set_ylabel('Next-Token Accuracy', fontsize=14)
    ax2.set_title('Next-Token Accuracy Across Samples', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Line plot saved to: {output_path}")


def run_utility_check(
    oracle_model_path: str,
    unlearned_model_path: str,
    retain_dataset_path: str,
    output_dir: str,
    max_samples: int = 200
):
    """
    Run utility check comparing oracle and unlearned models.

    Args:
        oracle_model_path: Path to oracle model
        unlearned_model_path: Path to unlearned model
        retain_dataset_path: Path to retain dataset
        output_dir: Directory to save results
        max_samples: Maximum samples to evaluate
    """
    print("=" * 80)
    print("UTILITY CHECK: ORACLE vs UNLEARNED MODEL")
    print("=" * 80)
    print(f"Oracle model: {oracle_model_path}")
    print(f"Unlearned model: {unlearned_model_path}")
    print(f"Retain dataset: {retain_dataset_path}")
    print(f"Max samples: {max_samples}")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    records = load_dataset(retain_dataset_path, max_samples)

    # Detect dataset type
    if 'client_name' in records[0]:
        dataset_type = 'medical_soap'
    else:
        dataset_type = 'wmdp'
    logger.info(f"Dataset type: {dataset_type}")

    # Load oracle model
    logger.info("Loading oracle model...")
    oracle_model, oracle_tokenizer = load_model_and_tokenizer(oracle_model_path)

    # Evaluate oracle model
    logger.info("Evaluating oracle model...")
    oracle_metrics = evaluate_model_utility(oracle_model, oracle_tokenizer, records, dataset_type)

    # Clean up oracle model
    del oracle_model
    torch.cuda.empty_cache()

    # Load unlearned model
    logger.info("Loading unlearned model...")
    unlearned_model, unlearned_tokenizer = load_model_and_tokenizer(unlearned_model_path)

    # Evaluate unlearned model
    logger.info("Evaluating unlearned model...")
    unlearned_metrics = evaluate_model_utility(unlearned_model, unlearned_tokenizer, records, dataset_type)

    # Clean up unlearned model
    del unlearned_model
    torch.cuda.empty_cache()

    # Compute summary statistics
    oracle_ppl_valid = [p for p in oracle_metrics['perplexities'] if math.isfinite(p)]
    unlearned_ppl_valid = [p for p in unlearned_metrics['perplexities'] if math.isfinite(p)]

    summary = {
        'oracle_model': oracle_model_path,
        'unlearned_model': unlearned_model_path,
        'retain_dataset': retain_dataset_path,
        'num_samples': len(records),
        'dataset_type': dataset_type,
        'oracle_avg_perplexity': np.mean(oracle_ppl_valid) if oracle_ppl_valid else float('inf'),
        'oracle_avg_accuracy': np.mean(oracle_metrics['accuracies']),
        'unlearned_avg_perplexity': np.mean(unlearned_ppl_valid) if unlearned_ppl_valid else float('inf'),
        'unlearned_avg_accuracy': np.mean(unlearned_metrics['accuracies']),
        'timestamp': datetime.now().isoformat()
    }

    # Save summary
    summary_path = os.path.join(output_dir, 'utility_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    # Create visualizations
    scatter_path = os.path.join(output_dir, 'utility_scatter.png')
    plot_scatter_comparison(oracle_metrics, unlearned_metrics, scatter_path)

    line_path = os.path.join(output_dir, 'utility_line.png')
    plot_line_comparison(oracle_metrics, unlearned_metrics, line_path)

    # Print summary
    print()
    print("=" * 80)
    print("UTILITY CHECK RESULTS")
    print("=" * 80)
    print(f"Samples evaluated: {len(records)}")
    print()
    print("Oracle Model:")
    print(f"  Average perplexity: {summary['oracle_avg_perplexity']:.4f}")
    print(f"  Average accuracy: {summary['oracle_avg_accuracy']:.4f}")
    print()
    print("Unlearned Model:")
    print(f"  Average perplexity: {summary['unlearned_avg_perplexity']:.4f}")
    print(f"  Average accuracy: {summary['unlearned_avg_accuracy']:.4f}")
    print()
    print("Utility Preservation:")
    if math.isfinite(summary['oracle_avg_perplexity']) and math.isfinite(summary['unlearned_avg_perplexity']):
        ppl_ratio = summary['unlearned_avg_perplexity'] / summary['oracle_avg_perplexity']
        print(f"  Perplexity ratio (unlearned/oracle): {ppl_ratio:.4f}")
    acc_diff = summary['unlearned_avg_accuracy'] - summary['oracle_avg_accuracy']
    print(f"  Accuracy difference (unlearned - oracle): {acc_diff:+.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model utility on retain set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--oracle_model',
        type=str,
        required=True,
        help='Path to oracle model (trained on full dataset)'
    )

    parser.add_argument(
        '--unlearned_model',
        type=str,
        required=True,
        help='Path to unlearned model (trained on full_minus_forget dataset)'
    )

    parser.add_argument(
        '--retain_dataset',
        type=str,
        required=True,
        help='Path to retain dataset'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/utility_check',
        help='Directory to save results (default: results/utility_check)'
    )

    parser.add_argument(
        '--max_samples',
        type=int,
        default=200,
        help='Maximum samples to evaluate (default: 200)'
    )

    args = parser.parse_args()

    run_utility_check(
        oracle_model_path=args.oracle_model,
        unlearned_model_path=args.unlearned_model,
        retain_dataset_path=args.retain_dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
