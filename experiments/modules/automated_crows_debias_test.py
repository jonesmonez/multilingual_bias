#!/usr/bin/env python3

import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

import json
import transformers
transformers.logging.set_verbosity_error()
import torch

current_dir = os.getcwd()
sys.path.append(current_dir)

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias

thisdir = os.path.dirname(os.path.realpath(__file__))

def test_crows_debias(
    path_to_crows: str,
    lang_debias: str,
    lang_eval: str,
    bias_type: str,
    bias_direction: str = None,
    projection_matrix: str = None,
    load_path: str = None,
    sample = False,
    model_name_or_path = "bert-base-uncased",
    pre_dir = os.path.realpath(os.path.join(thisdir, "..")),
    model_class = "SentenceDebiasBertForMaskedLM",
    save_result: bool = False,
    verbose: bool = True,
):

    if verbose:
        print("Running CrowS-Pairs benchmark:")
        print(f" - persistent_dir: {pre_dir}")
        print(f" - model: {model_class}")
        print(f" - model_name_or_path: {model_name_or_path}")
        print(f" - bias_direction: {bias_direction}")
        print(f" - projection_matrix: {projection_matrix}")
        print(f" - load_path: {load_path}")
        print(f" - bias_type: {bias_type}")
        print(f" - sample: {sample}")
        print(f" - lang_eval: {lang_eval}")
        print(f" - lang_debias: {lang_debias}")
    
    s=''
    kwargs = {}
    
    if bias_direction is not None:
        bias_direction = torch.load(bias_direction)
        if model_class == "DensrayDebiasBertForMaskedLM":
            print(bias_direction.size())
        kwargs["bias_direction"] = bias_direction
    
    if projection_matrix is not None:
        projection_matrix = torch.load(projection_matrix)
        kwargs["projection_matrix"] = projection_matrix
        k=projection_matrix
        s=k.replace('.pt','')
        s=s.replace('/','')

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model_class = getattr(models, model_class)(
        load_path or model_name_or_path, **kwargs
    )
    
    if _is_self_debias(model_class):
        model_class._model.eval()
    else:
        model_class.eval()
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    runner = CrowSPairsRunner(
        model=model_class,
        tokenizer=tokenizer,
        input_file=path_to_crows,
        bias_type=bias_type,
        is_generative=_is_generative(model_class),
        is_self_debias=_is_self_debias(model_class),
        sample=sample,
        verbose=verbose,
    )
    
    results, df_data_with_masks = runner()
    
    if save_result:
        experiment_id = generate_experiment_id(
            name="crows",
            model=model_class,
            model_name_or_path=model_name_or_path,
            bias_type=bias_type,
            sample=sample,
            lang_eval=lang_eval,
            lang_debias=lang_debias,
        )
        
        results_dir = f"{pre_dir}/results/automated_crows_debias_test"
        os.makedirs(results_dir, exist_ok=True)
        
        df_data_with_mask_probs.to_csv(f"{result_dir}/{experiment_id}{s}.csv")
        with open(f"{results_dir}/{experiment_id}.json", "w") as f:
            json.dump(results, f)

        print(f"Results saved to: {results_dir}/{experiment_id}.json")

    if verbose:
        print(f"Metric: {results}")
    
    return results

def _main():
    parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
    parser.add_argument(
        "--path",
        action="store",
        type=str,
        help="Path to the crows CSV.",
    )
    parser.add_argument(
        "--lang_debias",
        action="store",
        type=str,
        default='en',
        choices=['en', 'fr', 'nl', 'de','pl','ru','ca'],
        help="Language used to debias",
    )
    parser.add_argument(
        "--lang_eval",
        action="store",
        type=str,
        default='en',
        choices=['en', 'fr', 'nl', 'de','pl','ru','ca'],
        help="Language to evaluate on.",
    )
    parser.add_argument(
        "--bias_type",
        action="store",
        default="gender",
        choices=["gender", "race", "religion", "socioeconomic", "sexual-orientation", "age", "nationality", "disability", "physical-appearance"],
        help="Determines which CrowS-Pairs dataset split to evaluate against.",
    )
    parser.add_argument(
        "--bias_direction",
        action="store",
        type=str,
        help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
    )
    parser.add_argument(
        "--projection_matrix",
        action="store",
        type=str,
        help="Path to the file containing the pre-computed projection matrix for INLP.",
    )
    parser.add_argument(
        "--load_path",
        action="store",
        type=str,
        help="Path to saved ContextDebias, CDA, or Dropout model checkpoint.",
    )
    parser.add_argument(
        "--sample",
        action="store",
        type=str,
        default="false",
        choices=["true","false" ],
        help="Determines whether a sample of the dataset should be taken or not.",
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased","bert-base-multilingual-cased","bert-base-multilingual-uncased", "albert-base-v2", "roberta-base", "gpt2"],
        help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
        "model is instantiated.",
    )
    parser.add_argument(
        "--persistent_dir",
        action="store",
        type=str,
        default=os.path.realpath(os.path.join(thisdir, "..")),
        help="Directory where all persistent data will be stored.",
    )
    parser.add_argument(
        "--model_class",
        action="store",
        type=str,
        default="SentenceDebiasBertForMaskedLM",
        choices=[
            "DensrayDebiasBertForMaskedLM",
            "SentenceDebiasBertForMaskedLM",
            "SentenceDebiasAlbertForMaskedLM",
            "SentenceDebiasRobertaForMaskedLM",
            "SentenceDebiasGPT2LMHeadModel",
            "INLPBertForMaskedLM",
            "INLPmBertForMaskedLM",
            "INLPAlbertForMaskedLM",
            "INLPRobertaForMaskedLM",
            "INLPGPT2LMHeadModel",
            "CDABertForMaskedLM",
            "CDAAlbertForMaskedLM",
            "CDARobertaForMaskedLM",
            "CDAGPT2LMHeadModel",
            "DropoutBertForMaskedLM",
            "DropoutAlbertForMaskedLM",
            "DropoutRobertaForMaskedLM",
            "DropoutGPT2LMHeadModel",
            "SelfDebiasBertForMaskedLM",
            "SelfDebiasAlbertForMaskedLM",
            "SelfDebiasRobertaForMaskedLM",
            "SelfDebiasGPT2LMHeadModel",
            "SelfDebiasmBERTForMaskedLM",
        ],
        help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM). Typically, these "
        "correspond to a HuggingFace class.",
    )
    
    
    args = parser.parse_args()
    
    results = test_crows_debias(
        path_to_crows=args.path,
        lang_debias=args.lang_bias,
        lang_eval=args.lang_eval,
        bias_type=args.bias_type,
        bias_direction=args.bias_direction,
        projection_matrix=args.projection_matrix,
        load_path=args.load_path,
        sample=(args.sample == "true"),
        model_name_or_path=args.model_name_or_path,
        pre_dir=args.persistent_dir,
        model_class=args.model_class,
        save_result=True
    )
    
    print(f"Results: {results}")

if __name__ == "__main__":
    _main()
    
