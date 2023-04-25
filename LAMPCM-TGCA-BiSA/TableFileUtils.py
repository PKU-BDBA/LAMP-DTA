from sklearn.metrics import *
from scipy.stats import *
from matplotlib import pyplot as plt
import pandas as pd
import math
import os
import hashlib
from pathlib import *

def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def vc(series, to_dict=True, dropna=True):
    result = series.value_counts(dropna=dropna)
    if to_dict:
        return print(result.to_dict())
    print(result)

def str_hash(s, salt=''):
    return int(hashlib.sha256((str(s)+salt).encode('utf-8')).hexdigest(), 16)
    
def list_to_str(list):
    return [str(x) for x in list]

def df_split(list, ratios):
    results = []
    sum_value = sum(ratios)
    ratios = [x / sum_value for x in ratios]
    current = 0
    for ratio in ratios:
        results.append(list[int(len(list) * current):int(len(list) * (current + ratio))])
        current += ratio
    return results

def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result

def chunk_sample(list, n):
    result = []
    for i in range(1, n):
        result.append(list[math.floor(i / n * len(list))])
    return result


def chunk_to_batches(list, batch_size):
    result = []
    for i in range(0, len(list), batch_size):
        result.append(list[i:i + batch_size])
    return result

def ensure_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_file(filepath):
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)