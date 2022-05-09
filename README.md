# nlp_transformer_tutorial

[[_TOC_]]

## Introduction

learning note and tutorial for applying transformer to various NLP problems

The learning note is mainly based on huggingface transformer and dataset API.

## Notebooks

contains tutorials run step by step, suitable for colab or kaggle online jupyter notebook

notebooks can be found in learning_notes

## scripts

### Installing

scripts are script version of notebooks, dependencies are managed by `Poetry` and `VirtualEnv`. One need to install both to start, to install the dependencies:

```bash
poetry install
```

Get the virtualenv interpreter path if you need it:

```bash
poetry run which python
```

### How to run

```bash
poetry run poetry run transformer_tutorial_ch2
```

or depending on which chapter you would like to try, e.g.

```bash
poetry run poetry run transformer_tutorial_ch4
```
