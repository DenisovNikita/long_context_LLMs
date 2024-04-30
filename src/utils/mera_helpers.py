from definitions import *

def construct_prompt(info):
    return info['instruction'].format(**info['inputs'])