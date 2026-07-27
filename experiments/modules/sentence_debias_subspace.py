#!/usr/bin/env python3

import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

import torch
import transformers

from bias_bench.dataset import load_sentence_debias_data
from bias_bench.debias import (
    compute_gender_subspace,
    compute_race_subspace,
    compute_religion_subspace,
)
from bias_bench.model import models
from bias_bench.util import generate_experiment_id

def compute_bias_direction(
    persistent_dir: str,
    bias_type: str,
    batch_size: int = 32,
    lang_debias: str = "en",
    model_name: str = "BertModel",
    model_name_or_path: str = "bert-base-multilingual-uncased",
    save_result: bool = True,
    verbose: bool = False
):
    if verbose:
        print("Computing bias subspace:")
        print(f" - persistent_dir: {persistent_dir}")
        print(f" - model_name_or_path: {model_name_or_path}")
        print(f" - model: {model_name}")
        print(f" - bias_type: {bias_type}")
        print(f" - batch_size: {batch_size}")
        print(f" - language debias: {lang_debias}")

    data = load_sentence_debias_data(
        persistent_dir=persistent_dir,
        bias_type=bias_type,
        lang_debias=lang_debias
    )

    model = getattr(models, model_name)(model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    if model_name == "GPT2Model":
        tokenizer.pad_token = tokenizer.eos_token

    if bias_type == "gender":
        bias_direction = compute_gender_subspace(
            data, model, tokenizer, batch_size=batch_size
        )
    elif bias_type == "race-color":
        bias_direction = compute_race_subspace(
            data, model, tokenizer, batch_size=batch_size
        )
    elif bias_type == "religion":
        bias_direction = compute_religion_subspace(
            data, model, tokenizer, batch_size=batch_size
        )

    if save_result:
        experiment_id = generate_experiment_id(
            name="subspace",
            model=model_name,
            model_name_or_path=model_name_or_path,
            bias_type=bias_type,
            lang_debias=lang_debias,
        )

        print(f"Saving computed PCA components to: {persistent_dir}/results/subspace/{experiment_id}.pt.")
        os.makedirs(f"{persistent_dir}/results/subspace", exist_ok=True)
        torch.save(bias_direction, f"{persistent_dir}/results/subspace/{experiment_id}.pt")
        print(f"Saved to: {persistent_dir}/results/subspace/{experiment_id}.pt")

    return bias_direction

def _main():
    import argparse
    
    # When run as a script, use command line arguments like the original
    parser = argparse.ArgumentParser(description="Computes the bias subspace for SentenceDebias.")
    parser.add_argument(
        "--persistent_dir",
        action="store",
        type=str,
        default=os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
        help="Directory where all persistent data will be stored."
    )
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="BertModel",
        choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
        help="Model to compute the SentenceDebias subspace for."
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        type=str,
        default="bert-base-multilingual-uncased",
        choices=["bert-base-uncased",'bert-base-multilingual-uncased','bert-base-multilingual-cased',
                 "albert-base-v2", "roberta-base", "gpt2"],
        help="HuggingFace model name or path."
    )
    parser.add_argument(
        "--bias_type",
        action="store",
        type=str,
        choices=["gender", "religion", "race"],
        required=True,
        help="The type of bias to compute the bias subspace for."
    )
    parser.add_argument(
        "--lang_debias",
        action="store",
        type=str,
        default='en',
        choices=["nl", "en", "de", "fr","pl","ru","ca"],
        required=True,
        help="The language the bias is mitigated in."
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=32,
        help="Batch size to use while encoding."
    )

    args = parser.parse_args()

    compute_bias_direction(
        persistent_dir=args.persistent_dir,
        bias_type=args.bias_type,
        batch_size=args.batch_size,
        lang_debias=args.lang_debias,
        model_name=args.model,
        model_name_or_path=args.model_name_or_path,
        save_result=True
    )
    
if __name__ == "__main__":
    _main()