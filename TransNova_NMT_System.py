# Note: The model is developed using GoogleCollab

Install Libraries
"""

!pip install transformers sentencepiece sacrebleu accelerate gradio -q

"""IMPORT LIBRARIES

"""

import torch

from transformers import (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import pandas as pd
import sacrebleu
import time

"""LOAD MODEL 1 (M2M100)"""

print("Loading M2M100...")

m2m_tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M"
)

m2m_model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M"
)

m2m_model = m2m_model.to("cuda")

print("Loaded")

"""LOAD MODEL 2 (NLLB)"""

print("Loading NLLB...")

nllb_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

nllb_model = nllb_model.to("cuda")

print("Loaded")

"""CREATE LANGUAGE MAP"""

m2m_langs = {
    "English":"en",
    "French":"fr",
    "Hindi":"hi"
}

"""For NLLB:"""

nllb_langs = {
    "English":"eng_Latn",
    "French":"fra_Latn",
    "Hindi":"hin_Deva"
}

"""M2M TRANSLATION FUNCTION"""

def translate_m2m(
        text,
        src_lang,
        tgt_lang,
        max_length=128,
        beam_width=5,
        length_penalty=1.0
):

    m2m_tokenizer.src_lang = src_lang

    encoded = m2m_tokenizer(
        text,
        return_tensors="pt"
    ).to("cuda")

    generated = m2m_model.generate(
        **encoded,
        max_length=max_length,
        num_beams=beam_width,
        length_penalty=length_penalty,
        forced_bos_token_id=
        m2m_tokenizer.get_lang_id(tgt_lang)
    )

    return m2m_tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

"""TEST M2M"""

translate_m2m(
    "Hello, how are you?",
    "en",
    "fr"
)

"""NLLB TRANSLATION FUNCTION"""

def translate_nllb(
        text,
        src_lang,
        tgt_lang,
        max_length=128,
        beam_width=5,
        length_penalty=1.0
):

    nllb_tokenizer.src_lang = src_lang

    encoded = nllb_tokenizer(
        text,
        return_tensors="pt"
    ).to("cuda")

    generated = nllb_model.generate(
        **encoded,
        max_length=max_length,
        num_beams=beam_width,
        length_penalty=length_penalty,
        forced_bos_token_id=
        nllb_tokenizer.convert_tokens_to_ids(
            tgt_lang
        )
    )

    return nllb_tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

"""TEST NLLB"""

translate_nllb(
    "Hello, how are you?",
    "eng_Latn",
    "fra_Latn"
)

"""LANGUAGE VALIDATION"""

def validate_language(lang, model):

    if model=="M2M":

        valid = [
            "en",
            "fr",
            "hi"
        ]

    else:

        valid = [
            "eng_Latn",
            "fra_Latn",
            "hin_Deva"
        ]

    return lang in valid

"""HANDLE EMPTY INPUT"""

def validate_text(text):

    if len(text.strip())==0:
        return False

    return True

# test
validate_text("")

"""HANDLE LONG INPUT"""

def validate_length(text):

    if len(text) > 1000:
        return False

    return True

"""PARAMETER EXPERIMENT"""

translate_m2m(
    text="Artificial Intelligence is changing the world.",
    src_lang="en",
    tgt_lang="fr",
    beam_width=2
)

translate_m2m(
    text="Artificial Intelligence is changing the world.",
    src_lang="en",
    tgt_lang="fr",
    beam_width=5
)

translate_m2m(
    text="Artificial Intelligence is changing the world.",
    src_lang="en",
    tgt_lang="fr",
    beam_width=8
)

"""BLEU SCORE FUNCTION"""

from sacrebleu import corpus_bleu

def calculate_bleu(reference, prediction):

    bleu = corpus_bleu(
        [prediction],
        [[reference]]
    )

    return round(bleu.score, 2)

# Experimental Analysis
news_text = "The government announced a new economic policy."

tech_text = "Transformer models use self attention mechanisms."

conv_text = "How are you doing today?"

# French references
news_reference = "Le gouvernement a annoncé une nouvelle politique économique."

tech_reference = "Les modèles Transformer utilisent des mécanismes d'auto-attention."

conv_reference = "Comment allez-vous aujourd'hui ?"

# Generate M2M100 Translations
news_pred_m2m = translate_m2m(
    news_text,
    "en",
    "fr"
)

tech_pred_m2m = translate_m2m(
    tech_text,
    "en",
    "fr"
)

conv_pred_m2m = translate_m2m(
    conv_text,
    "en",
    "fr"
)

# Generate NLLB Translations
news_pred_nllb = translate_nllb(
    news_text,
    "eng_Latn",
    "fra_Latn"
)

tech_pred_nllb = translate_nllb(
    tech_text,
    "eng_Latn",
    "fra_Latn"
)

conv_pred_nllb = translate_nllb(
    conv_text,
    "eng_Latn",
    "fra_Latn"
)

# Calculate BLEU Scores
# M2M100
news_bleu_m2m = calculate_bleu(
    news_reference,
    news_pred_m2m
)

tech_bleu_m2m = calculate_bleu(
    tech_reference,
    tech_pred_m2m
)

conv_bleu_m2m = calculate_bleu(
    conv_reference,
    conv_pred_m2m
)

# NLLB
news_bleu_nllb = calculate_bleu(
    news_reference,
    news_pred_nllb
)

tech_bleu_nllb = calculate_bleu(
    tech_reference,
    tech_pred_nllb
)

conv_bleu_nllb = calculate_bleu(
    conv_reference,
    conv_pred_nllb
)

# BLEU Comparison Table
bleu_results = pd.DataFrame({

    "Text Type":[
        "News",
        "Technical",
        "Conversational"
    ],

    "M2M100 BLEU":[
        news_bleu_m2m,
        tech_bleu_m2m,
        conv_bleu_m2m
    ],

    "NLLB BLEU":[
        news_bleu_nllb,
        tech_bleu_nllb,
        conv_bleu_nllb
    ]
})

bleu_results

# Save Results
bleu_results.to_csv(
    "bleu_scores.csv",
    index=False
)

print("BLEU results saved successfully.")

"""PERFORMANCE BENCHMARK"""

def benchmark_translation(
        model_function,
        text,
        src_lang,
        tgt_lang
):

    torch.cuda.empty_cache()

    start_time = time.time()

    translation = model_function(
        text,
        src_lang,
        tgt_lang
    )

    end_time = time.time()

    translation_time = round(
        end_time - start_time,
        3
    )

    return translation, translation_time

# BENCHMARK BOTH MODELS
benchmark_text = """
Artificial Intelligence and Machine Learning are transforming industries
by enabling intelligent decision-making and automation.
"""

# M2M100 Benchmark
m2m_translation, m2m_time = benchmark_translation(
    translate_m2m,
    benchmark_text,
    "en",
    "fr"
)

# NLLB Benchmark
nllb_translation, nllb_time = benchmark_translation(
    translate_nllb,
    benchmark_text,
    "eng_Latn",
    "fra_Latn"
)

# Performance Comparison Table
performance_results = pd.DataFrame({

    "Model":[
        "M2M100",
        "NLLB"
    ],

    "Translation Time (seconds)":[
        m2m_time,
        nllb_time
    ]
})

performance_results

"""PARAMETER TUNING"""

# Beam Width Experiment
beam_results = []

for beam in [2, 4, 6, 8]:

    translated = translate_m2m(
        "Artificial Intelligence is changing the world.",
        "en",
        "fr",
        beam_width=beam
    )

    beam_results.append([
        beam,
        translated
    ])

# Beam Width Table
beam_df = pd.DataFrame(
    beam_results,
    columns=[
        "Beam Width",
        "Translation"
    ]
)

beam_df

"""FINAL BLEU SCORE TABLE"""

final_bleu_results = pd.DataFrame({

    "Text Type":[
        "News",
        "Technical",
        "Conversational"
    ],

    "M2M100 BLEU":[
        news_bleu_m2m,
        tech_bleu_m2m,
        conv_bleu_m2m
    ],

    "NLLB BLEU":[
        news_bleu_nllb,
        tech_bleu_nllb,
        conv_bleu_nllb
    ]
})

final_bleu_results

"""SAVE RESULTS"""

final_bleu_results.to_csv(
    "bleu_results.csv",
    index=False
)

performance_results.to_csv(
    "performance_results.csv",
    index=False
)

beam_df.to_csv(
    "beam_width_results.csv",
    index=False
)

print("All result files saved.")

"""BUILD INTERACTIVE GRADIO GUI"""

import gradio as gr
def translate_interface(
        text,
        model_choice,
        max_length,
        beam_width,
        length_penalty
):

    if len(text.strip()) == 0:
        return "Please enter text."

    if model_choice == "M2M100":

        return translate_m2m(
            text,
            "en",
            "fr",
            max_length,
            beam_width,
            length_penalty
        )

    else:

        return translate_nllb(
            text,
            "eng_Latn",
            "fra_Latn",
            max_length,
            beam_width,
            length_penalty
        )

# CREATE GUI
demo = gr.Interface(

    fn=translate_interface,

    inputs=[

        gr.Textbox(
            lines=5,
            label="Enter Text"
        ),

        gr.Dropdown(
            ["M2M100","NLLB"],
            label="Choose Model"
        ),

        gr.Slider(
            32,
            512,
            value=128,
            label="Max Length"
        ),

        gr.Slider(
            1,
            10,
            value=5,
            label="Beam Width"
        ),

        gr.Slider(
            0.5,
            2.0,
            value=1.0,
            label="Length Penalty"
        )
    ],

    outputs=[
        gr.Textbox(
            label="Translated Output"
        )
    ],

    title="Neural Machine Translation System",

    description="""
    Compare M2M100 and NLLB Transformer Models
    """
)

# LAUNCH GUI
demo.launch(
    debug=True
)
