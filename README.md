# CS-4063: Natural Language Processing — Assignment 3
## Transformers + RAG: Grounded Review Understanding and Explanation

**Author:** Muhammad Noor | i23-2520 | FAST NUCES  
**Project Objective:** A complete, from-scratch implementation of a Transformer-based RAG pipeline for sentiment analysis and human-like explanation generation.

---
## 1. Project Architecture

This system implements a modular three-stage pipeline to process Amazon product reviews without using high-level Transformer libraries.

### A. Part A: The Encoder (Multi-Task Learning)
- **Architecture:** Encoder-only Transformer built from scratch (Positional Encoding, Multi-Head Attention, LayerNorm).
- **Task:** Jointly classifies **Sentiment** (3 classes) and **Review Length** (3 classes).
- **Evaluation:** Includes loss curves, accuracy metrics, and confusion matrices for both validation and test sets.

### B. Part B: The Retrieval Module
- **Architecture:** Vector-based semantic search.
- **Task:** Uses the trained Encoder to generate fixed-dimensional embeddings for 45,000 training reviews.
- **Logic:** Implements Cosine Similarity to find the Top-K most relevant "context" reviews to help ground the generation process.

### C. Part C: The Decoder (RAG-Enabled Generation)
- **Architecture:** Causal Transformer Decoder with Masked Multi-Head Attention.
- **Task:** Generates text explanations using **Retrieval-Augmented Generation**. The model "looks" at the retrieved context to produce more coherent and grounded explanations.
- **Ablation Study:** Quantifies the benefit of RAG by comparing model perplexity against a non-retrieval baseline.

---

## 2. Comprehensive Directory Structure

### Root Files
- `i23-2520_Assignment3_DS-A.ipynb` — **Core Project Notebook**. Contains the complete code, training loops, and results.
- `i23-2520_Assignment3_DS-A_data-processor.py` — Preprocessing script to clean and sample the raw Amazon dataset.
- `i23-2520_Assignment3_DS-A_cleaned-dataset.csv` — The final processed dataset used for all experiments.
- `report.tex` — LaTeX source for the final research report.
- `requirements.txt` — Python dependencies (PyTorch, Pandas, Matplotlib, etc.).
- `NLP_Assignment3.pdf` — Official assignment description and grading rubric.

### `models/` (Checkpoints)
- `encoder_checkpoint.pth` — Saved weights/state for the Sentiment Encoder.
- `decoder_checkpoint.pth` — Saved weights for the RAG-grounded Decoder.
- `baseline_checkpoint.pth` — Weights for the ablation study baseline.
*Note: All training loops feature **Auto-Resumption** from these checkpoints.*

###  `results/` (Artifacts)
- `train_embeddings.pth` — The persistent vector database used for Part B retrieval.
- `encoder_loss_curve.png` / `decoder_loss_curve.png` — Visualizations of training convergence.
- `rag_ablation_perplexity.png` — Comparison plot proving RAG's effectiveness.
- `bonus_ui_output.png` — Exported image from the interactive Summary UI.

### `Dataset/`
- Contains the raw Amazon `.json.gz` files.

---

## 3. Execution Instructions

Follow these steps in order to replicate the results:

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Processing:**
    Run the processor to generate the cleaned CSV from raw data.
    ```bash
    python i23-2520_Assignment3_DS-A_data-processor.py
    ```

3.  **Run the Notebook:**
    Open `i23-2520_Assignment3_DS-A.ipynb` and **Run All Cells**.
    - The notebook will check for existing checkpoints in `models/`.
    - If found, it will **Resume Training** or skip to evaluation, saving you hours of compute time.

---

## 4. Key Implementation Features
- **From-Scratch Implementation:** Every core block (Multi-Head Attention, Scaled Dot-Product, Positional Encodings) is written in raw PyTorch without using `nn.Transformer`.
- **RAG Grounding:** Unlike standard GPT models, this decoder uses a specific prompt structure:  
  `Review: [Input] | Sent: [Predicted] | Feat: [Length] | Context: [Retrieved Match] -> Explanation:`
- **Data Cleanup:** Uses an advanced pipeline (HTML stripping, contraction expansion, and length bucketing) to ensure high-quality embeddings.
- **Visual Analytics:** Beyond just metrics, the notebook generates Attention Heatmaps and a Premium Interactive UI for qualitative testing.

---
## 5. Rubric Compliance Checklist

- [x] **Part A (30 Marks):** Scratch Transformer, Multi-task classification, Evaluation plots.
- [x] **Part B (15 Marks):** Efficient embedding storage, Cosine Similarity, Proof of retrieval quality.
- [x] **Part C (35 Marks):** Masked Decoder, RAG prompt logic, Baseline vs RAG Ablation Study.
- [x] **Bonus (5 Marks):** Heatmaps and stylized Interactive Prediction UI.

---
© 2026 Muhammad Noor (i23-2520) | FAST NUCES
