#!/usr/bin/env python3

import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

import json
import transformers

current_dir = os.getcwd()
sys.path.append(current_dir)

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

thisdir = os.path.dirname(os.path.realpath(__file__))

def test_crows(
    path: str,
    bias_type: str,
    sample = False,
    model_name_or_path = "bert-base-uncased",
    pre_dir = os.path.realpath(os.path.join(thisdir, "..")),
    model_class = "BertModel",
    save_result: bool = True
):

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {pre_dir}")
    print(f" - model_name_or_path: {model_name_or_path}")
    print(f" - bias_type: {bias_type}")
    print(f" - sample: {sample}")

    # Load model and tokenizer.
    model = getattr(models, model_class)(model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file=path,
        bias_type=bias_type,
        is_generative=_is_generative(model),
        sample=sample,
   )
    results, df_data_with_masks = runner()

    if save_result:
        experiment_id = generate_experiment_id(
            name="automated_test",
            model=args.model_class,
            model_name_or_path=args.model_name_or_path,
            bias_type=args.bias_type,
            sample=(args.sample == "true"),
            seed=args.seed,
            lang_eval=args.lang_eval,
        )

        results_dir = f"{args.persistent_dir}/results/automated_test"
        os.makedirs(results_dir, exist_ok=True)

        with open(f"{results_dir}/{experiment_id}.json", "w") as f:
            json.dump(results, f)

        print(f"Results saved to: {results_dir}/{experiment_id}.json")

    print(f"Metric: {results}")

    print(results)
    return results

def _main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
    parser.add_argument(
        "--persistent_dir",
        action="store",
        type=str,
        default=os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
        help="Directory where all persistent data will be stored.",
    )
    parser.add_argument(
        "--model_class",
        action="store",
        type=str,
        default="BertModel",
        choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model",
                "BertForMaskedLM", "AlbertForMaskedLM", "RobertaForMaskedLM", "GPT2LMHeadModel"],
        help="Model class to use (e.g., BertModel).",
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        type=str,
        default="bert-base-multilingual-uncased",
        choices=["bert-base-uncased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased",
                "albert-base-v2", "roberta-base", "gpt2"],
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--bias_type",
        action="store",
        type=str,
        default="gender",
        choices=["gender", "race", "religion", "socioeconomic", "sexual-orientation", "age", "nationality", "disability", "physical-appearance"],
        help="Type of bias to evaluate.",
    )
    parser.add_argument(
        "--sample",
        action="store",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to use a sample of the dataset.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=None,
        help="Random seed for seeded datasets (e.g., for crows_en_US_s0.csv).",
    )
    parser.add_argument(
        "--path",
        action="store",
        type=str,
        required=True,
        help="File to evaluate (.csv).",
    )

    args = parser.parse_args()

    results = test_crows(
        path=path,
        bias_type=args.bias_type,
        sample=(args.sample == "true"),
        model_name_or_path=args.model_name_or_path,
        pre_dir=args.persistent_dir,
        model_class=args.model_class,
        save_result=True
    )

    print(f"Results: {results}")
    

if __name__ == "__main__":
    _main()