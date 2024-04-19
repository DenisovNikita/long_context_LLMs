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