# learning notes and jupyter notebooks

## chapter 1:

A introduction of the huggingface pipeline
- how to load model
- how to apply model

Different NLP problems and huggingface models available in model hub

## chapter 2:

Text classification or sentimental analysis

- take the last logits of CLS position as input
- train a logitical-regression/mlp on top of it
- For classification, one widely used is a AutoModelForSequenceClassification model
  since the last layer is not trained, when loading this model, you will see a warning
