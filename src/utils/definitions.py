import numpy as np
import pandas as pd
import datetime
import json
import requests
from bs4 import BeautifulSoup
from soup2dict import convert
import time
from IPython.display import clear_output
from pathlib import Path
from tqdm import tqdm, trange
import sys
import os
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM # , LlamaTokenizer
import typing as tp
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import peft
import torch
import subprocess
from datasets import load_from_disk, load_dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score


BOOKS = [
    "and_quiet_flows_the_don",
    "anna_karenina",
    "crime_and_punishment",
    "dead_souls",
    "doctor_zhivago",
    "evenings_on_a_farm_near_dikanka",
    "idiot",
    "oblomov",
    "the_brothers_karamazov",
    "the_dawns_here_are_quiet",
    "the_gulag_archipelago",
    "the_heart_of_a_dog",
    "the_master_and_margarita",
    "the_white_guard",
    "war_and_peace",
]
DIPLOMA_DIR_PATH = Path(__file__).parent.joinpath("../../..").resolve()
REPOSITOTY_DIR_PATH = DIPLOMA_DIR_PATH.joinpath("long_context_LLMs")
ARTIFACTS_DIR_PATH = REPOSITOTY_DIR_PATH.joinpath("artifacts")
METRICS_DIR_PATH = ARTIFACTS_DIR_PATH.joinpath("metrics")
DATASETS_DIR_PATH = ARTIFACTS_DIR_PATH.joinpath("datasets")

LLAMA_2_7B = "llama-2-7b"
LLAMA_3_8B = "llama-3-8b"
FALCON_7B = "falcon-7b"
BLOOM = "bloom"
FLAN_T5_XLL = "flan-t5-xxl"
FRED_T5_1_7B = "fred-t5-1.7b"
PHI_2 = "phi"
DOLLY_V2_7B = "dolly-v2-7b"
DECI_LM_7B = "DeciLM-7b"
SAIGA_MISTRAL_7B_LORA = "saiga_mistral_7b_lora"
VICUNA_7B = "vicuna-7b"
VIKHR_7B = "vikhr-7b"
RU_LONGFORMER_LARGE_4096 = "ru-longformer-large-4096"
MISTRAL_7B = "mistral-7b"
MIXTRAL_8X7B = "mixtral-8x7b"
GEMMA_7B = "gemma-7b"

MODELS = {
    LLAMA_2_7B,
    LLAMA_3_8B,
    FALCON_7B,
    BLOOM,
    FLAN_T5_XLL,
    FRED_T5_1_7B,
    PHI_2,
    DOLLY_V2_7B,
    DECI_LM_7B,
    SAIGA_MISTRAL_7B_LORA,
    VICUNA_7B,
    VIKHR_7B,
    RU_LONGFORMER_LARGE_4096,
    MISTRAL_7B,
    MIXTRAL_8X7B,
    GEMMA_7B,
}

HUGGINFACE_BASELINE_MODELS = {
    LLAMA_2_7B,
    MISTRAL_7B,
}

HUGGINFACE_RUSSIAN_MODELS = {
    SAIGA_MISTRAL_7B_LORA,
    VIKHR_7B,
    FRED_T5_1_7B,
    RU_LONGFORMER_LARGE_4096,
}

HUGGINGFACE_MODEL_TO_REPO = {
    LLAMA_2_7B: "meta-llama/Llama-2-7b-hf",
    # LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B", # banned for me personal
    FALCON_7B: "tiiuae/falcon-7b",
    BLOOM: "bigscience/bloom",
    FLAN_T5_XLL: "google/flan-t5-xxl",
    FRED_T5_1_7B: "ai-forever/FRED-T5-1.7B",
    PHI_2: "microsoft/phi-2",
    DOLLY_V2_7B: "databricks/dolly-v2-7b",
    DECI_LM_7B: "Deci/DeciLM-7B",
    SAIGA_MISTRAL_7B_LORA: "IlyaGusev/saiga_mistral_7b_lora",
    VICUNA_7B: "lmsys/vicuna-7b-v1.5",
    VIKHR_7B: "Vikhrmodels/Vikhr-7b-0.1",
    RU_LONGFORMER_LARGE_4096: "kazzand/ru-longformer-large-4096",
    MISTRAL_7B: "mistralai/Mistral-7B-v0.1",
    MIXTRAL_8X7B: "mistralai/Mixtral-8x7B-v0.1",
    GEMMA_7B: "google/gemma-7b",
}

HUGGINGFACE_NAME_TO_DATASET = {
    "mera": {
        "repo": "ai-forever/MERA",
        "subsets": ["rummlu", "ruopenbookqa"],
        "splits": ["public_test", "train"],
    }
}

# there are quantized models
# OLLAMA_MODEL_TO_NAME = {
#     LLAMA_2_7B: "llama2:latest",
#     LLAMA_3_8B: "llama3:latest",
#     VICUNA_7B: "vicuna"
# }
