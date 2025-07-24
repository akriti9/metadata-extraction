# metadata-extraction
#  LLaMA-ArXiv Metadata Extraction

This project fine-tunes the [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3-8B-Instruct) model using the QLoRA method to extract **structured metadata** from arXiv research papers, such as:

- `id` (arXiv ID)
- `title`
- `authors`
- `abstract`
- `doi`

The solution leverages instruction tuning and parameter-efficient fine-tuning (QLoRA) to enable large-scale information extraction using limited hardware.

---

##  Overview

This repository contains two main components:

1. **Data Preparation** (`prepare_data_for_finetuning.py`)  
   - Cleans, truncates, and formats research papers into LLaMA-style instruction prompts.

2. **Fine-Tuning Pipeline** (`finetune_llama.py`)  
   - Fine-tunes LLaMA-3.1-8B-Instruct using Hugging Face’s `Trainer` and `peft` (QLoRA), logging all metrics, saving checkpoints, and final model.

---

##  Project Structure

```
llama-arxiv-metadata-extraction/
│
├── prepare_data_for_finetuning.py   # Data cleaning + instruction-format converter
├── finetune_llama.py                # Full training pipeline using QLoRA + Hugging Face
├── processed_data/                  # Output of prepared training/val/test JSON files
│   ├── train_conversations.json
│   ├── validation_conversations.json
│   └── test_conversations.json
├── results/                         # Final model, logs, and metrics
│   ├── training_metrics.json
│   ├── latest_checkpoint_info.json
│   └── logs/
├── training.log                     # Runtime logs
└── README.md                        # You're reading this :)
```

---

##  Getting Started

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

> Also ensure:
> - You have access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face.
> - Your machine has a GPU (for CPU-only training, expect significant slowdown).
> - `bitsandbytes` is installed correctly with PyTorch CUDA support.

---

### 2️⃣ Prepare the Data

Ensure your raw dataset is a JSON list with entries like:

```json
{
  "id": "2301.12345",
  "title": "Your paper title",
  "authors": ["Author One", "Author Two"],
  "abstract": "This paper explores...",
  "article_text": "Full paper content...",
  "doi": "10.1000/example.doi"
}
```

Then run:

```bash
python prepare_data_for_finetuning.py
```

This will generate:
- `train_conversations.json`
- `validation_conversations.json`
- `test_conversations.json`

All in instruction format, stored in `./processed_data/`.

---

### 3️⃣ Fine-Tune LLaMA

```bash
python finetune_llama.py
```

This will:
- Load LLaMA-3.1-8B-Instruct in 4-bit using QLoRA
- Train it using the conversation-style data
- Log losses and save checkpoints in `./results/`
- Save final model in `./finetuned_model/`

Optional: Set `WANDB_API_KEY` env variable to enable Weights & Biases logging.

---

##  Instruction Format (LLaMA-style)

```txt
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at extracting structured metadata from research papers...
<|eot_id|><|start_header_id|>user<|end_header_id|>
Extract information from this research paper:

<full paper text>

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{
  "id": "2301.12345",
  "title": "Deep Learning for Health",
  "authors": ["John Doe", "Jane Smith"],
  "abstract": "This paper explores...",
  "doi": "10.1000/xyz123"
}
<|eot_id|>
```

---

##  Configuration Highlights

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Quantization:** 4-bit QLoRA (`bnb_4bit`)
- **LoRA Rank:** 16, Alpha: 32
- **Batch Size:** 2 (with gradient accumulation)
- **Max Tokens:** 3072 per input
- **Eval Steps:** 200
- **Epochs:** 5
- **Scheduler:** Cosine LR

---

##  Output Example

Once training is complete, outputs include:
- Trained model in `./finetuned_model/`
- Training logs: `training.log`
- Metric file: `training_metrics.json`
- Best model checkpoint (auto-loaded)

---

## 📃 Acknowledgments

This work was done as part of a **research internship at IISER Bhopal**, supervised by **Dr. Tanmay Basu**, Head of the Department of Data Science and Engineering.  
The project aimed to explore **NLP for evidence synthesis** in biomedical literature.

---

## 📎 License

This repository is for **academic and research** use only. Please cite appropriately if reused.



---

##  Requirements (for reference)

Create a `requirements.txt` with:

```txt
transformers>=4.39.0
datasets
peft
accelerate
bitsandbytes
tqdm
wandb
```
