# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import os
import random
import torch
from datasets import load_dataset, load_from_disk


# Preserve server defaults; configure offline datasets before launching Python.
LOCAL_C4_TRAIN = os.environ.get("WANDA_C4_TRAIN", "/data/DataSet/llm_dataset/c4/c4train")
LOCAL_C4_VALIDATION = os.environ.get("WANDA_C4_VALIDATION", "/data/DataSet/llm_dataset/c4/c4validation")
LOCAL_WIKITEXT2_TRAIN = os.environ.get("WANDA_WIKITEXT2_TRAIN", "/data/DataSet/llm_dataset/wikitext2/wikitext2_train")
LOCAL_WIKITEXT2_TEST = os.environ.get("WANDA_WIKITEXT2_TEST", "/data/DataSet/llm_dataset/wikitext2/wikitext2_test")

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Use server-local train/test datasets; evaluation must not access the Hub.
    traindata = load_from_disk(LOCAL_WIKITEXT2_TRAIN)
    testdata = load_from_disk(LOCAL_WIKITEXT2_TEST)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load the server-local C4 datasets; calibration must not access the Hub.
    traindata = load_from_disk(LOCAL_C4_TRAIN)
    valdata = load_from_disk(LOCAL_C4_VALIDATION)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
