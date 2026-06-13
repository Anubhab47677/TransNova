# TransNova

## Transformer-Based Neural Machine Translation System

TransNova is a Neural Machine Translation (NMT) system developed using Transformer architectures and Hugging Face models. The project performs multilingual translation and compares the performance of two state-of-the-art translation models:

* Facebook M2M100 (418M)
* Facebook NLLB-200 Distilled (600M)

---

## Project Objectives

* Build a Neural Machine Translation System
* Compare two Transformer-based translation models
* Perform translation quality evaluation using BLEU Score
* Analyze translation performance and speed
* Conduct parameter tuning experiments
* Develop an interactive user interface using Gradio

---

## Models Used

### M2M100

A multilingual many-to-many translation model developed by Meta AI.

### NLLB-200

No Language Left Behind (NLLB) is a multilingual translation model supporting more than 200 languages.

---

## Features

* Multilingual Translation
* Transformer-Based Architecture
* BLEU Score Evaluation
* Performance Benchmarking
* Parameter Tuning
* Language Validation
* Empty Input Handling
* Long Input Handling
* Interactive Gradio Interface

---

## Languages Supported

### M2M100

* English (en)
* French (fr)
* Hindi (hi)

### NLLB

* English (eng_Latn)
* French (fra_Latn)
* Hindi (hin_Deva)

---

## Experimental Analysis

The models are evaluated on:

1. News Text
2. Technical Text
3. Conversational Text

Evaluation Metric:

* BLEU Score

Performance Metric:

* Translation Time

---

## Parameter Tuning

The following parameters are analyzed:

* Beam Width
* Maximum Sequence Length
* Length Penalty

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* SacreBLEU
* Pandas
* Gradio
* Google Colab

---

## Installation

Install dependencies:

```bash
pip install transformers sentencepiece sacrebleu accelerate gradio pandas torch
```

---

## Running the Project

Open the notebook in Google Colab and run all cells sequentially.

The Gradio interface will launch automatically after execution.

---

## Outputs Generated

* BLEU Score Comparison Table
* Performance Comparison Table
* Beam Width Analysis
* CSV Result Files
* Interactive Translation Interface

---

## Future Improvements

* Additional language support
* Real-time translation API
* Speech-to-Text integration
* Text-to-Speech integration
* Translation memory system

---

## Author

Anubhab Samantaray

Natural Language Processing with Transformers Project