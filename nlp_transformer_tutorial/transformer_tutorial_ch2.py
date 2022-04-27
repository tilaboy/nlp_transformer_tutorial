from datasets import list_datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, pipeline, AutoModel, TFAutoModel
from transformers import DistilBertTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import tensorflow as tf
import torch
import math
import umap
import umap.plot
from torch.nn.functional import cross_entropy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score

# global variables:
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(batch): # Place model inputs on the GPU
	  inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in tokenizer.model_input_names
	  }
    # Extract last hidden states
	  with torch.no_grad():
		    last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
	  return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def forward_pass_with_label(batch): # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
        # Place outputs on CPU for compatibility with other dataset columns
        return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}

def label_int2str(row):
	  return emotions["train"].features["label"].int2str(row)

all_data_sets = list_datasets()
print(f'in total {len(all_data_sets)}\nfirst 10: {all_data_sets[:10]}')

emotions = load_dataset("emotion")
train_set = emotions['train']
print('columns', train_set.column_names)
print('nr sampels', len(train_set))
print('features', train_set.features)
print('first sample', train_set[0])

emotions.set_format(type='pandas')

df = emotions['train'][:]
df.head()

label_mapper = emotions["train"].features["label"].int2str
df["label_name"] = df["label"].apply(label_mapper)
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

emotions.reset_format()

text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'])
print('bert tokens:', tokens)
print('orig tokens:', tokenizer.convert_tokens_to_string(tokens))
print(encoded_text)
print(tokenizer.vocab_size, tokenizer.model_max_length, tokenizer.model_input_names, tokenizer.max_model_input_sizes)

for i in range(1, 3):
    print(tokenize(emotions["train"][:i]))

emotions_encoded = emotions.map(tokenize, batched = True, batch_size=None)
print(emotions_encoded["train"].column_names)

model_ckpt = "distilbert-base-uncased"
print(f'using device {device}')

text = "this is a test"
inputs_pt = tokenizer(text, return_tensors="pt")
inputs_tf = tokenizer(text, return_tensors="tf")
print('pt', inputs_pt)
print('tf', inputs_tf)

inputs = {k:v.to(device) for k,v in inputs_pt.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
outputs.last_hidden_state.size()

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
emotions_hidden["train"].column_names

X_train = np.array(emotions_hidden["train"][:4000]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"][:1000]["hidden_state"])
y_train = np.array(emotions_hidden["train"][:4000]["label"])
y_valid = np.array(emotions_hidden["validation"][:1000]["label"])
print(X_train.shape, X_valid.shape)

X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = umap.UMAP(n_components=2, metric="cosine").fit(X_scaled)
label_names = emotions["train"].features["label"].names
label_mapper = {i:label_names[i] for i in range(len(label_names))}
print(label_mapper)
umap.plot.points(mapper, labels=np.array([label_mapper[label] for label in y_train]))

print('type', type(emotions_hidden['train']))
print(emotions_hidden['train'].shape, emotions_hidden['train'].num_columns, emotions_hidden['train'].num_rows)
print(emotions_hidden['train'].column_names)
print(emotions_hidden['train'].features)
for i in range(20):
    item_tokens = tokenizer.convert_ids_to_tokens([token_id for token_id in emotions_hidden['train'][i]['input_ids'] if token_id != 0])
    item_label = emotions_hidden['train'][i]['label'].tolist()
    print(tokenizer.convert_tokens_to_string(item_tokens),
          ' => ',
          item_label,
          label_mapper[item_label])

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print('dummy classifier score:', dummy_clf.score(X_valid, y_valid))

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
print('logistic regression score:', lr_clf.score(X_valid, y_valid))

y_dummy = dummy_clf.predict(X_valid)
plot_confusion_matrix(y_dummy, y_valid, label_names)

y_clf = lr_clf.predict(X_valid)
plot_confusion_matrix(y_clf, y_valid, label_names)

num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error"
)
trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train();

preds_output = trainer.predict(emotions_encoded["validation"])
print('metrics from predition output', preds_output.metrics)
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, emotions_hidden["validation"]["label"], label_names)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map( forward_pass_with_label, batched=True, batch_size=16)
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))
df_test.sort_values("loss", ascending=False).head(10)

df_test.sort_values("loss", ascending=True).head(10)

custom_tweet = "I saw a movie today and it was really good."
input_custom_tweet = tokenizer(custom_tweet, return_tensors="pt")
print(input_custom_tweet)
inputs = {k:v.to(device) for k,v in input_custom_tweet.items()}
with torch.no_grad():
    preds = model(**inputs)
print(preds['logits'])

num_sum = sum([math.exp(ele) for ele in preds['logits'][0].tolist()])
plt.bar(label_names, [math.exp(ele)/num_sum for ele in preds['logits'][0].tolist()], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()


tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
tokenizer_columns = tokenizer.model_input_names
tf_train_dataset = emotions_encoded["train"].to_tf_dataset(columns=tokenizer_columns,
                                                           label_cols=["label"],
                                                           shuffle=True,
                                                           batch_size=batch_size,
                                                           collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf"))
tf_eval_dataset = emotions_encoded["validation"].to_tf_dataset(columns=tokenizer_columns,
                                                               label_cols=["label"],
                                                               shuffle=False,
                                                               batch_size=batch_size,
                                                               collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf"))
tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=tf.metrics.SparseCategoricalAccuracy()
                 )
tf_model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=2)

tf_preds_output = tf_model.predict(tf_eval_dataset)
tf_preds = np.argmax(tf_preds_output['logits'], axis=1)
plot_confusion_matrix(tf_preds, emotions_encoded["validation"]['label'], label_names)

input_custom_tweet = tokenizer(custom_tweet, return_tensors="tf")
print(input_custom_tweet)
with torch.no_grad():
    preds = tf_model(**input_custom_tweet)
print(preds['logits'])

num_sum = sum([math.exp(ele) for ele in preds['logits'][0]])
plt.bar(label_names, [math.exp(ele)/num_sum for ele in preds['logits'][0]], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
