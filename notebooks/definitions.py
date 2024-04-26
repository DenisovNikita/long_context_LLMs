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
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Any, Dict, List
import matplotlib.pyplot as plt


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
DIPLOMA_DIR_PATH = Path(__file__).parent.joinpath("../..").resolve()
REPOSITOTY_DIR_PATH = DIPLOMA_DIR_PATH.joinpath("long_context_LLMs")
ARTIFACTS_DIR_PATH = REPOSITOTY_DIR_PATH.joinpath("artifacts")

HUGGINGFACE_MODEL_TO_REPO = {
    "llama-2-7b": "meta-llama/Llama-2-7b",
    # "llama-3-8b": "meta-llama/Meta-Llama-3-8B", # banned for me personal
    "falcon-7b": "tiiuae/falcon-7b",
    "bloom": "bigscience/bloom",
    "flan-t5-xxl": "google/flan-t5-xxl",
    "fred-t5-1.7b": "ai-forever/FRED-T5-1.7B",
    "phi": "microsoft/phi-2",
    "dolly-v2-7b": "databricks/dolly-v2-7b",
    "DeciLM-7b": "Deci/DeciLM-7B",
    "saiga_mistral_7b_lora": "IlyaGusev/saiga_mistral_7b_lora",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "vikhr-7b": "Vikhrmodels/Vikhr-7b-0.1",
    "ru-longformer-large-4096": "kazzand/ru-longformer-large-4096",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "gemma-7b": "google/gemma-7b",
}
