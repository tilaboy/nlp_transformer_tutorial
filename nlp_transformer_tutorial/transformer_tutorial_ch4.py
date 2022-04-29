import os
import logging
import math

from datasets import load_dataset, DatasetDict
from datasets import get_dataset_config_names, concatenate_datasets
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoConfig, TrainingArguments
from transformers import XLMRobertaConfig, Trainer
from transformers import DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def check_data_set():
    xtreme_subsets = get_dataset_config_names("xtreme")
    print(f"XTREME has {len(xtreme_subsets)} configurations")
    panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")]
    print(f'nr of languages in PAN dataset: {len(panx_subsets)}')
    print([set_name[-2:] for set_name in panx_subsets])


def fetch_data(langs):
    fracs = [0.629, 0.229, 0.084, 0.059]
    # Return a DatasetDict if a key doesn't exist
    panx_ch = defaultdict(DatasetDict)
    for lang, frac in zip(langs, fracs):
        # Load monolingual corpus
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        # Shuffle and downsample each split according to spoken proportion
        for split in ds:
            nr_to_select = int(frac * ds[split].num_rows / 4)
            print(f'{lang}-{split}: {nr_to_select} out of {ds[split].num_rows}')
            panx_ch[lang][split] = ( ds[split].shuffle(seed=0).select(range(nr_to_select)))
    return panx_ch

def create_tag_names(batch):
    return {'ner_tags_str': [tags.int2str(idx) for idx in batch['ner_tags']]}


def main():
    langs = ["de", "fr", "it", "en"]
    panx_ch = fetch_data(langs)
