import os
import logging
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

from . import LOGGER

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def log_probs_from_logits(logits, labels):
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label


def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits( output.logits[:, :-1, :], labels[:, 1:])
    seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_txt = "this zoo has introduced"
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
    iterations = []
    n_steps = 8
    choices_per_step = 5
    with torch.no_grad():
        for _ in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)
            # Select logits of the first batch and the last token and apply softmax
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
            # Store tokens with highest probabilities
            for choice_idx in range(choices_per_step):
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = ( f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)" )
                iteration[f"Choice {choice_idx+1}"] = token_choice
            # Append predicted next token to input
            input_ids = torch.cat([input_ids, sorted_ids[0].reshape(1, 1)], dim=-1)
            iterations.append(iteration)
    print(pd.DataFrame(iterations))

    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
    print(tokenizer.decode(output[0]))

    max_length = 128
    input_txt = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."""
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)

    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
    logp = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_greedy[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False)
    logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_beam[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False, no_repeat_ngram_size=2)
    logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_beam[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_temp = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=2.0, top_k=0)
    logp = sequence_logprob(model, output_temp, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_temp[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_temp = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.5, top_k=0)
    logp = sequence_logprob(model, output_temp, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_temp[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_topk = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50)
    logp = sequence_logprob(model, output_topk, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_topk[0]))
    print(f"\nlog-prob: {logp:.2f}")

    output_topp = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.90)
    logp = sequence_logprob(model, output_topp, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_topp[0]))
    print(f"\nlog-prob: {logp:.2f}") 

    output_all = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=True, temperature=0.5, top_k=50, top_p=0.9)
    logp = sequence_logprob(model, output_all, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_all[0]))
    print(f"\nlog-prob: {logp:.2f}")

if __name__ == '__main__':
    main()
