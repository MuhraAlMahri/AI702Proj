# Contradiction-Aware LLMs

## Project Overview
Large Language Models (LLMs) show impressive capabilities but struggle with recognizing and resolving self-contradictory instructions.  
This project focuses on improving LLMs' ability to detect contradictions and provide clarifications, increasing their reliability and trustworthiness.

We evaluate two instruction-tuned models:
- **LLaMA-3.2-1B-Instruct**
- **Qwen-2.5-3B-Instruct**

We explore two strategies:
- **Few-shot prompting** (3, 6, 12 examples) during inference (In-Context Learning).
- **Parameter-efficient fine-tuning** using **LoRA** and **QLoRA**.

Our experiments are based on the SCI-Benchmark (self-contradictory instructions, language-language-3 subset).

---

## Repository Structure
- `inference_finetuned_llama.py`: Inference using fine-tuned LLaMA model.
- `inference_baseline_llama.py`: Inference using baseline LLaMA model.
- `inference_baseline_qwen.py`: Inference using baseline Qwen model.
- `requirements.txt`: List of Python dependencies required to run the code.
- `README.md`: This documentation file.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers accelerate datasets
```

Recommended environment: Python 3.8+, GPU (for faster inference), CUDA toolkit installed.

---

## How to Run

To run inference on a model:

```bash
python inference_finetuned_llama.py
```

For baseline (no fine-tuning):

```bash
python inference_baseline_llama.py
```

For Qwen baseline:

```bash
python inference_baseline_qwen.py
```

Ensure the models are properly available in the script paths or downloaded from Hugging Face.

---

## Dataset

We use the **SCI-Benchmark (language-language-3 subset)**, specifically focusing on prompts containing mutually exclusive instructions.

Example input:
```
Instruction 1: Summarize the paragraph in a detailed, long way.
Instruction 2: Summarize the paragraph in one short sentence.
```

---

## Evaluation Metrics

We use two metrics to evaluate model performance:
- **BLEU Score**: Measures similarity between generated output and expected response.
- **Keyword Matching**: Measures if outputs include contradiction-related keywords ("contradiction", "conflict", "inconsistent", etc.).

---

## Key Findings

| Model                   | BLEU Score | Keyword Matching (%) |
|--------------------------|------------|-----------------------|
| Zero-shot LLaMA baseline  | 0.0545     | 1.60                  |
| Fine-tuned LLaMA (LoRA)   | 0.0682     | 19.40                 |

Fine-tuning with LoRA significantly improves explicit contradiction recognition compared to zero-shot performance.

---

## Credits

- Models: [LLaMA-3.2-1B-Instruct](https://huggingface.co/meta-llama) and [Qwen-2.5-3B-Instruct](https://huggingface.co/Qwen)
- Fine-tuning Techniques: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314)
- Dataset: [SCI-Benchmark](https://arxiv.org/abs/2404.13208)

---

## Authors
- Chenxi Wang
- Muhra AlMahri
- Ayesha AlHammadi
(MBZUAI, Group ID: G-07)
