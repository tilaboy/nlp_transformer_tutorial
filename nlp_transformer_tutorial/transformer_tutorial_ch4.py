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
from . import LOGGER

# env variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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


def create_tag_names(tags):
    def apply_tag_names(batch):
        return {'ner_tags_str': [tags.int2str(idx) for idx in batch['ner_tags']]}
    return apply_tag_names


def dataset_summary(panx_dataset):
    split2freqs = defaultdict(Counter)
    for split, dataset in panx_de.items():
        for row in dataset["ner_tags_str"]:
            for tag in row:
                if tag.startswith("B"):
                    tag_type = tag.split("-")[1]
                    split2freqs[split][tag_type] += 1
    df_split_summary = pd.DataFrame.from_dict(split2freqs, orient="index")
    print(df_split_summary.describe())


def tag_text(text, tags, model, tokenizer):
    # Get tokens with special characters
    token_obj = tokenizer(text, return_tensors="pt")
    tokens = token_obj.tokens()
    # Encode the sequence into IDs
    input_ids = token_obj.input_ids.to(device)
    # Get predictions as distribution over 7 possible classes
    outputs = model(input_ids)[0]
    # Take argmax to get most likely class per token
    predictions = torch.argmax(outputs, dim=2)
    # Convert to DataFrame
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["Tokens", "Tags"])


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.roberta = RobertaModel(config, add_pooling_layer = False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None, **kwargs):
        #Use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # Return model output object
        return TokenClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, attentions = outputs.attentions)


def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx] [seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
            labels_list.append(example_labels)
            preds_list.append(example_preds)
    return preds_list, labels_list

def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'])

def get_train_args(model_name, batch_size, num_epochs, logging_steps):
    return TrainingArguments(output_dir=model_name,
                              log_level="error",
                              num_train_epochs=num_epochs,
                              per_device_train_batch_size=batch_size,
                              per_device_eval_batch_size=batch_size,
                              evaluation_strategy="epoch",
                              save_steps=1e6,
                              weight_decay=0.01,
                              disable_tqdm=False,
                              logging_steps=logging_steps,
                              push_to_hub=False)

def train_model(checkpoint_name, config, model_name, train_set, validation_set, tokenizer):
    num_epochs = 3
    batch_size = 24
    logging_steps = len(panx_de_encoded["train"]) // batch_size
    model_name = f"{xlmr_model_name}-finetuned-panx-de"

    def model_init():
        return XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)

    trainer = Trainer(model_init=model_init,
                      args=get_train_args(model_name, batch_size, num_epochs, logging_steps),
                      data_collator=DataCollatorForTokenClassification(xlmr_tokenizer),
                      compute_metrics=compute_metrics,
                      train_dataset=panx_de_encoded["train"],
                      eval_dataset=panx_de_encoded["validation"],
                      tokenizer=xlmr_tokenizer)
    trainer.train()
    return trainer


def plot_confusion_matrix(y_preds, y_true, labels, output_name):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.savefig(output_name)

def get_samples(df):
    for _, row in df.iterrows():
        labels, predictions, tokens, losses = [], [], [], []
        print(row['attention_mask'], len(row['attention_mask']))
        for i, mask in enumerate(row['attention_mask']):
            if i == 0 or i == len(row['attention_mask']):
                continue
            labels.append(row['labels'][i])
            predictions.append(row['predicted_label'][i])
            tokens.append(row['input_tokens'][i])
            losses.append(round(row['loss'][i], 3))
        yield pd.DataFrame({'label': labels, 'prediction': predictions, 'tokens': tokens, 'losses': losses}).T

def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]

def evaluate_lang_performance(trainer, data_set):
    encoded_dataset = encode_panx_dataset(dataset)
    return get_f1_score(trainer, encoded_dataset["test"])

def train_on_subset(model_name, dataset, num_samples, training_args):
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_samples))
    valid_ds = dataset["validation"]
    test_ds = dataset["test"]
    get_train_args(model_name, 24, 3, len(train_ds) // 24),

    trainer = Trainer(model_init=model_init,
                      args=training_args,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      train_dataset=train_ds,
                      eval_dataset=valid_ds,
                      tokenizer=xlmr_tokenizer)
    trainer.train()
    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message="Training completed!")
    f1_score = get_f1_score(trainer, test_ds)
    return pd.DataFrame.from_dict( {"num_samples": [len(train_ds)], "f1_score": [f1_score]})

def concatenate_splits(corpora):
    multi_corpus = DatasetDict()
    for split in corpora[0].keys():
        multi_corpus[split] = concatenate_datasets( [corpus[split] for corpus in corpora]).shuffle(seed=42)
    return multi_corpus


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_data_set()
    langs = ["de", "fr", "it", "en"]
    panx_ch = fetch_data(langs)
    LOGGER.info('apply ner tag converting')
    tags = panx_ch["de"]["train"].features["ner_tags"].feature
    print(tags)
    panx_de = panx_ch["de"].map(create_tag_names)
    print(panx_de["train"][0])

    dataset_summary(panx_ch["de"])

    LOGGER.info('BERT V.S. Roberta')
    text = "Jack Sparrow loves New York!"
    bert_model_name = "bert-base-cased"
    xlmr_model_name = "xlm-roberta-base"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

    bert_tokens = bert_tokenizer(text).tokens()
    xlmr_tokens = xlmr_tokenizer(text).tokens()
    print('bert tokens', bert_tokens)
    print('xlmr tokens', xlmr_tokens)
    #improve from roberta tokenzier, when replace _, get original input text
    #"".join(xlmr_tokens).replace(u"\u2581", " ")
    #"".join(xlmr_tokens).replace("▁", " ")

    LOGGER.info('load the xlmr model, and test with example')
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}
    xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, num_labels=tags.num_classes, id2label=index2tag, label2id=tag2index)
    xlmr_model = XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)

    input_ids = xlmr_tokenizer.encode(text, return_tensors="pt")
    df_example_encoded = pd.DataFrame([xlmr_tokens, input_ids[0].numpy()], index=["Tokens", "Input IDs"])
    print(df_example_encoded.describe())

    xlmr_trainer = train_model()
    print('tags to predict', tags)
    text_de = "Jeff Dean ist ein Informatiker bei Google in Kalifornien"
    tag_text(text_de, tags, trainer.model, xlmr_tokenizer)
    text_fr = "Jeff Dean est informaticien chez Google en Californie"
    tag_text(text_fr, tags, trainer.model, xlmr_tokenizer)


    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

    def forward_pass_with_label(batch):
        # Convert dict of lists to list of dicts suitable for data collator
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        # Pad inputs and labels and put all tensors on device
        batch = data_collator(features)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            # Pass data through model
            output = trainer.model(input_ids, attention_mask)
            # logit.size: [batch_size, sequence_length, classes]
            # Predict class with largest logit value on classes axis
            predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()
        # Calculate loss per token after flattening batch dimension with view
        loss = cross_entropy(output.logits.view(-1, 7), labels.view(-1), reduction="none")
        # Unflatten batch dimension and convert to numpy array
        loss = loss.view(len(input_ids), -1).cpu().numpy()
        return {"loss":loss, "predicted_label": predicted_label}

    valid_set = panx_de_encoded["validation"]
    valid_set = valid_set.map(forward_pass_with_label, batched=True, batch_size=32)
    index2tag[-100] = "IGN"
    df = valid_set.to_pandas()
    df["input_tokens"] = df["input_ids"].apply( lambda x: xlmr_tokenizer.convert_ids_to_tokens(x))
    df["predicted_label"] = df["predicted_label"].apply( lambda x: [index2tag[i] for i in x])
    df["labels"] = df["labels"].apply( lambda x: [index2tag[i] for i in x])
    df['loss'] = df.apply( lambda x: x['loss'][:len(x['input_ids'])], axis=1)
    df['predicted_label'] = df.apply( lambda x: x['predicted_label'][:len(x['input_ids'])], axis=1)
    print(df.head(5))

    df_tokens = df.apply(pd.Series.explode)
    df_tokens = df_tokens.query("labels != 'IGN'")
    df_tokens["loss"] = df_tokens["loss"].astype(float).round(2)
    print(df_tokens.head(20))

    print( df_tokens.groupby("input_tokens")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)
    # Get rid of multi-level columns
    .sort_values(by="sum", ascending=False)
    #.reset_index()
    .round(2)
    .head(10)
    .T
    )

    print( df_tokens.groupby("labels")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)
    .sort_values(by="mean", ascending=False)
    #.reset_index()
    .round(2)
    .T
    )

    plot_confusion_matrix(df_tokens["labels"], df_tokens["predicted_label"], tags.names)
    df['total_loss'] = df['loss'].apply(sum)
    df_tmp = df.sort_values(by='total_loss', ascending=False).head(5)

    for sample in get_samples(df_tmp):
        display(sample)
    df_tmp = df.loc[df["input_tokens"].apply(lambda x: u"\u2581(" in x)].head(5)
    for sample in get_samples(df_tmp):
        display(sample)

    f1_scores = defaultdict(dict)
    for lang in ['de', 'fr', 'it', 'en']:
        f1_scores["de"][lang] = evaluate_lang_performance(trainer, panx_ch[lang])
        print(f"F1-score of [de] model on [{lang}] dataset: {f1_scores['de'][lang]:.3f}")


    metrics_df = pd.DataFrame()
    for num_samples in [200, 500, 1000, 1500, 2000]:
        metrics_df = metrics_df.append( train_on_subset(panx_fr_encoded, num_samples), ignore_index=True)
    fig, ax = plt.subplots()
    ax.axhline(f1_scores["de"]["fr"], ls="--", color="r")
    metrics_df.set_index("num_samples").plot(ax=ax)
    plt.legend(["Zero-shot from de", "Fine-tuned on fr"], loc="lower right")
    plt.ylim((0, 1))
    plt.xlabel("Number of Training Samples")
    plt.ylabel("F1 Score")
    plt.savefig('zero_shot_fr.png')

    panx_all_encoded = concatenate_splits([panx_de_encoded, panx_fr_encoded])
    training_args.logging_steps = len(panx_all_encoded["train"]) // batch_size
    training_args.push_to_hub = False
    training_args.output_dir = "xlm-roberta-base-finetuned-panx-de-fr"
    trainer = Trainer(model_init=model_init,
                      args=training_args,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      tokenizer=xlmr_tokenizer,
                      train_dataset=panx_all_encoded["train"],
                      eval_dataset=panx_all_encoded["validation"])
    trainer.train()

    for lang in ['de', 'fr', 'it', 'en']:
        f1_scores["de_fr"][lang] = evaluate_lang_performance(lang, trainer)
        print(f"F1-score of [de_fr] model on [{lang}] dataset: {f1_scores['de_fr'][lang]:.3f}")

    corpora = [panx_de_encoded]
    # Exclude German from iteration
    for lang in langs[1:]:
        training_args.output_dir = f"xlm-roberta-base-finetuned-panx-{lang}"
        # Fine-tune on monolingual corpus
        ds_encoded = encode_panx_dataset(panx_ch[lang])
        metrics = train_on_subset(ds_encoded, ds_encoded["train"].num_rows)
        # Collect F1-scores in common dict
        f1_scores[lang][lang] = metrics["f1_score"][0]
        # Add monolingual corpus to list of corpora to concatenate
        corpora.append(ds_encoded)

    corpora_encoded = concatenate_splits(corpora)
    training_args.logging_steps = len(corpora_encoded["train"]) // batch_size
    training_args.output_dir = "xlm-roberta-base-finetuned-panx-all"
    trainer = Trainer(model_init=model_init, args=training_args, data_collator=data_collator, compute_metrics=compute_metrics, tokenizer=xlmr_tokenizer, train_dataset=corpora_encoded["train"], eval_dataset=corpora_encoded["validation"])
    trainer.train()

    for idx, lang in enumerate(langs):
        f1_scores["all"][lang] = get_f1_score(trainer, corpora[idx]["test"])
    scores_data = {"de": f1_scores["de"], "each": {lang: f1_scores[lang][lang] for lang in langs}, "all": f1_scores["all"]}
    f1_scores_df = pd.DataFrame(scores_data).T.round(4)
    f1_scores_df.rename_axis(index="Fine-tune on", columns="Evaluated on", inplace=True)
    print(f1_scores_df)


if __name__ == '__main__':
    main()
