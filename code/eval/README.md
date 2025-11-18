# Evaluation Scripts for Data Extraction Attacks

This directory contains evaluation scripts for measuring memorization and data extraction vulnerabilities in language models, based on the paper "Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs".

## Scripts

### 1. `memorization_baseline.py`

Computes memorization baseline by prompting models with partial sequences (prefix p + question q) without any guidance.

**Purpose**: Establish baseline extraction scores for comparison.

**Usage**:
```bash
python memorization_baseline.py \
    --model_path models/llama-3.1-8b-medical-oracle \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_path results/memorization_baseline.jsonl \
    --max_samples 100 \
    --prefix_length 20
```

**Key Features**:
- Supports both medical SOAP notes and WMDP datasets
- Generates completions using prompt + question format
- Computes exact match, token overlap, and prefix match metrics
- Saves detailed results for further analysis

### 2. `utility_check.py`

Compares model utility (accuracy/perplexity) on retain set between oracle and unlearned models.

**Purpose**: Verify that unlearning doesn't degrade model performance on retained data.

**Usage**:
```bash
python utility_check.py \
    --oracle_model models/llama-3.1-8b-medical-oracle \
    --unlearned_model models/llama-3.1-8b-medical-unlearned \
    --retain_dataset dataset/med_synthetic_full_minus_forget01.json \
    --output_dir results/utility_check \
    --max_samples 200
```

**Key Features**:
- Computes perplexity and next-token accuracy
- Generates scatter plots comparing oracle vs unlearned models
- Creates line plots showing metric trends across samples
- Saves summary statistics and visualizations

### 3. `eval.py`

Comprehensive evaluation with ROUGE-L and A-ESR metrics.

**Purpose**: Measure data extraction success using standard NLP metrics.

**Metrics**:
- **ROUGE-L**: Longest common subsequence-based similarity
  - Precision: LCS length / candidate length
  - Recall: LCS length / reference length
  - F1: Harmonic mean of precision and recall

- **A-ESR(θ)**: Approximate Extraction Success Rate with threshold θ
  - A-ESR(0.9): Proportion of samples with ROUGE-L F1 ≥ 0.9
  - A-ESR(1.0): Proportion of samples with ROUGE-L F1 = 1.0 (exact match)

**Usage**:

Single model evaluation:
```bash
python eval.py \
    --model_path models/llama-3.1-8b-medical-oracle \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_path results/eval_results.jsonl \
    --max_samples 100
```

Compare multiple models:
```bash
python eval.py \
    --compare \
    --model_paths models/llama-3.1-8b-medical-oracle models/llama-3.1-8b-medical-unlearned \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_dir results/comparison
```

**Key Features**:
- ROUGE-L computation (precision, recall, F1)
- A-ESR(0.9) and A-ESR(1.0) metrics
- Model comparison mode
- Detailed per-sample results

### 4. `test_eval_metrics.py`

Unit tests for evaluation metrics with hard-coded toy sequences.

**Purpose**: Verify correctness of ROUGE-L and A-ESR implementations.

**Usage**:
```bash
# Using pytest
python -m pytest test_eval_metrics.py -v

# Or directly
python test_eval_metrics.py
```

**Test Coverage**:
- ROUGE-L exact match (should be 1.0)
- ROUGE-L no overlap (should be 0.0)
- ROUGE-L partial overlap
- ROUGE-L with subset sequences
- ROUGE-L case insensitivity
- ROUGE-L edge cases (empty strings)
- A-ESR with various thresholds
- A-ESR boundary conditions
- Integration tests

## Installation

Install required dependencies:
```bash
pip install torch transformers numpy matplotlib seaborn tqdm
```

For testing:
```bash
pip install pytest
```

## Example Workflow

### 1. Compute Memorization Baseline

```bash
# For medical oracle model
python memorization_baseline.py \
    --model_path models/llama-3.1-8b-medical-oracle \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_path results/medical_baseline.jsonl \
    --max_samples 100

# For WMDP oracle model
python memorization_baseline.py \
    --model_path models/llama-3.1-8b-wmdp \
    --dataset_path dataset/wmdp_bio_forget01.json \
    --output_path results/wmdp_baseline.jsonl \
    --max_samples 100
```

### 2. Check Utility Preservation

```bash
python utility_check.py \
    --oracle_model models/llama-3.1-8b-medical-oracle \
    --unlearned_model models/llama-3.1-8b-medical-unlearned \
    --retain_dataset dataset/med_synthetic_full_minus_forget01.json \
    --output_dir results/utility_check \
    --max_samples 200
```

### 3. Comprehensive Evaluation

```bash
# Evaluate oracle model
python eval.py \
    --model_path models/llama-3.1-8b-medical-oracle \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_path results/oracle_eval.jsonl \
    --max_samples 100

# Evaluate unlearned model
python eval.py \
    --model_path models/llama-3.1-8b-medical-unlearned \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_path results/unlearned_eval.jsonl \
    --max_samples 100

# Compare models
python eval.py \
    --compare \
    --model_paths models/llama-3.1-8b-medical-oracle models/llama-3.1-8b-medical-unlearned \
    --dataset_path dataset/med_synthetic_forget01.json \
    --output_dir results/comparison
```

### 4. Run Unit Tests

```bash
python test_eval_metrics.py
```

## Output Files

### Memorization Baseline (`memorization_baseline.jsonl`)

```json
{
  "model_path": "models/llama-3.1-8b-medical-oracle",
  "dataset_path": "dataset/med_synthetic_forget01.json",
  "num_samples": 100,
  "avg_exact_match": 0.15,
  "avg_token_overlap": 0.65,
  "avg_prefix_match": 0.72
}
{
  "sample_id": 0,
  "prompt": "Complete the following medical SOAP note:\n...",
  "target": "...",
  "generated": "...",
  "exact_match": 0.0,
  "token_overlap": 0.68,
  "prefix_match": 0.75
}
```

### Utility Check (`utility_summary.json`)

```json
{
  "oracle_model": "models/llama-3.1-8b-medical-oracle",
  "unlearned_model": "models/llama-3.1-8b-medical-unlearned",
  "num_samples": 200,
  "oracle_avg_perplexity": 5.23,
  "oracle_avg_accuracy": 0.78,
  "unlearned_avg_perplexity": 5.45,
  "unlearned_avg_accuracy": 0.76
}
```

### Evaluation Results (`eval_results.jsonl`)

```json
{
  "summary": {
    "model_path": "models/llama-3.1-8b-medical-oracle",
    "num_samples": 100,
    "avg_rouge_l_f1": 0.72,
    "a_esr_0.9": 0.25,
    "a_esr_1.0": 0.08
  }
}
{
  "sample_id": 0,
  "rouge_l_precision": 0.85,
  "rouge_l_recall": 0.82,
  "rouge_l_f1": 0.84
}
```

## Metrics Interpretation

### ROUGE-L F1 Score
- **1.0**: Perfect extraction (exact match)
- **≥ 0.9**: Very high similarity (near-perfect extraction)
- **0.5-0.9**: Moderate similarity (partial extraction)
- **< 0.5**: Low similarity (failed extraction)

### A-ESR(0.9)
- Measures proportion of "near-perfect" extractions
- High A-ESR(0.9) indicates strong memorization vulnerability

### A-ESR(1.0)
- Measures proportion of exact extractions
- High A-ESR(1.0) indicates severe memorization vulnerability

### Perplexity
- Lower is better (model is more confident)
- Significant increase indicates utility degradation

### Next-Token Accuracy
- Higher is better
- Should remain similar between oracle and unlearned models

## Notes

- All scripts support both medical SOAP notes and WMDP datasets
- Automatic dataset type detection based on record structure
- GPU recommended for model inference
- Results are saved incrementally to prevent data loss
- Visualizations saved as high-resolution PNG files (300 DPI)

## Citation

If you use these evaluation scripts, please cite the original paper:

```bibtex
@article{unlearned2025,
  title={Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs},
  author={...},
  journal={arXiv preprint arXiv:2505.24379},
  year={2025}
}
```
