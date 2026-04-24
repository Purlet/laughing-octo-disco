# laughing-octo-disco
An AI model that predicts which response users will prefer in a head-to-head battle between chatbots powered by different large language models. This allows the model to learn task-specific patterns while retaining the general language knowledge it gained during pre-training.

The dataset comprises conversations from the chatbot arena, a site where different LLMs generate answers to user prompts.
Dataset: https://www.kaggle.com/competitions/llm-classification-finetuning/data

Multiple approaches were used to tackle this problem. These are:
  - BERT-CLS Dual Encoder — Chatbot Preference Classification
  - TF-IDF coupled with LSTM, FNN, CNN, and Logistic Regression
  - LLM-as-a-judge
  - Reinforcement Learning with Bradley Terry and Roberta Reward Models

## Models and their files
  - BERT-CLS Dual Encoder : Bert-cls-dual-encoder.ipynb
  - TF-IDF/LSTM and Logistic Regression:logReg_and_BLSTM.ipynb
  - TF-IDF/FNN : TF-IDF-1.ipynb
  - TF-IDF/CNN : TF-IDF-CNN.ipynb
  - LLM-as-a-judge : LLM_as_a_judge.ipynb
  - Bradley Terry: Reinforcement_Model.ipynb
  - roBERTa: RLHF.ipynb

## Requirements

```bash
pip install torch transformers scikit-learn pandas bitsandbytes datasets accelerate sentencepiece
pip install sentencepiece
pip install scikit-learn
pip install tensorflow
pip install torch
pip install groq
```

## Usage o

1. Mount your dataset to `/content/drive/MyDrive/Colab Notebooks/train.csv`
2. Run cells in order — tokenization → embedding extraction → classifier training
3. Loss and accuracy curves are plotted automatically after training

## Hardware

Developed and tested on Google Colab with an NVIDIA T4 GPU (16 GB VRAM). CPU inference is not recommended due to the high computational cost of samples above 15,000.
