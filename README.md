# BERT-CLS Dual Encoder — Chatbot Preference Classification

A pairwise preference classifier built on frozen BERT embeddings, trained on the LMSys Chatbot Arena dataset to predict which of two AI-generated responses a human would prefer.

## Overview

This model encodes each (prompt, response) pair independently through BERT-base-uncased, extracts the [CLS] token embedding from the final hidden layer, and concatenates both 768-dimensional vectors into a single 1,536-dimensional representation. A lightweight feedforward classifier is trained on top of these frozen embeddings to predict one of three outcomes: response_a preferred, response_b preferred, or tie.

BERT's weights are never updated — only the classifier head is trained (~330K parameters), making this approach fast, memory-efficient, and a strong baseline for the pairwise preference task.

## Model Architecture

## Dataset

- **Source:** LMSys Chatbot Arena
- **Size:** 15,000 samples (truncated from full dataset due to memory constraints)
- **Split:** 70% train / 30% validation, stratified by class
- **Classes:** `0` = model_a wins, `1` = model_b wins, `2` = tie

## Results

| Metric | Value |
|---|---|
| Final validation accuracy | 43.58% |
| Trainable parameters | ~330K |
| Base model | bert-base-uncased |

## Requirements

```bash
pip install torch transformers scikit-learn pandas bitsandbytes
```

## Usage

1. Mount your dataset to `/content/drive/MyDrive/Colab Notebooks/train.csv`
2. Run cells in order — tokenization → embedding extraction → classifier training
3. Loss and accuracy curves are plotted automatically after training

## Hardware

Developed and tested on Google Colab with an NVIDIA T4 GPU (16 GB VRAM). CPU inference is not recommended due to the cost of BERT's 12-layer forward pass across 15,000 samples.
