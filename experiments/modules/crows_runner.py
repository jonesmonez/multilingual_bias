import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import torch
import transformers
transformers.logging.set_verbosity_error()
import pandas as pd
from pathlib import Path
from functools import lru_cache
from bias_bench.model import models
from bias_bench.util import _is_generative, _is_self_debias

from experiments.modules.experiment_name import filename
from bias_bench.benchmark.crows.crows import CrowSPairsRunner
from bias_bench.benchmark.crows.crows_batched import CrowSPairsBatchedRunner


class CrowSPairsRunnerWrapper:
    def __init__(
        self,
        model_name_or_path: str = "bert-base-multilingual-uncased",
        model_class_base: str = "BertForMaskedLM",
        model_class_debias: str = "SentenceDebiasBertForMaskedLM",
        device: str | None = None,
        save_result: bool = False,
        save_path: Path = Path("results/"),
        verbose: bool = False,
        batched: bool = False,
    ):  
        self.batched = batched
        
        self.model_name_or_path = model_name_or_path
        self.model_class_base = model_class_base
        self.model_class_debias = model_class_debias
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_base = None
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        
        self.save_result = save_result
        self.save_path = save_path
        self.verbose = verbose

    def _build_base_model(self):
        self.model_base = getattr(models, self.model_class_base)(self.model_name_or_path)
        self.model_base.eval()
        
        self.model_base.to(self.device)
        
        self._is_generative = _is_generative(self.model_base)
        
    def save_json_output(self, to_save_data, method, bias_type, lang_debias, lang_eval):
        name = filename(method, bias_type, lang_debias, lang_eval, self.model_name_or_path) + ".json"
        save_to = self.save_path / name
        self.save_path.mkdir(parents=True, exist_ok=True)
        with save_to.open(mode="w", encoding="UTF-8") as f:
            json.dump(to_save_data, f)
        if self.verbose:
            print(f"Results saved to: \"{save_to.resolve()}\"")
            
    def csv_output_path(self, method, bias_type, lang_debias, lang_eval):
        name = filename(method, bias_type, lang_debias, lang_eval, self.model_name_or_path) + ".csv"
        save_to = self.save_path / name
        self.save_path.mkdir(parents=True, exist_ok=True)
        return save_to
    
    @staticmethod
    def create_runner(batched: bool, **kwargs):
        if batched:
            return CrowSPairsBatchedRunner(**kwargs)
        return CrowSPairsRunner(**kwargs)

    def run_base(
        self,
        path_to_crows: str,
        lang_eval: str,
        bias_type: str,
        sample = False,
    ):
        if self.model_base is None:
            self._build_base_model()
        
        if self.verbose:
            print("Running CrowS-Pairs benchmark:")
            print(f" - save_path: {self.save_path}")
            print(f" - model_name_or_path: {self.model_name_or_path}")
            print(f" - bias_type: {bias_type}")
            print(f" - sample: {sample}")

        runner = self.create_runner(
            batched=self.batched,
            model=self.model_base,
            tokenizer=self.tokenizer,
            input_file=path_to_crows,
            lang_eval=lang_eval,
            bias_type=bias_type,
            is_generative=_is_generative(self.model_base),
            sample=sample,
            verbose=self.verbose,
        )
        results, df_data_with_masks = runner()

        if self.save_result:
            self.save_json_output(results, "crows-base", bias_type, None, lang_eval)

        if self.verbose:
            print(f"Metric: {results}")

        return results, df_data_with_masks
    
    @lru_cache(maxsize=4)
    def _build_debias_model(
        self,
        model_class_str: str,
        load_path: str | None,
        bias_direction: str | None,
        projection_matrix: str | None,
    ):
        DebiasCls = getattr(models, model_class_str)

        kwargs = {}
        if bias_direction is not None:
            kwargs["bias_direction"] = torch.load(bias_direction, map_location="cpu")
        if projection_matrix is not None:
            kwargs["projection_matrix"] = torch.load(projection_matrix, map_location="cpu")

        weight_source = load_path if load_path is not None else self.model_name_or_path

        debias_model = DebiasCls(weight_source, **kwargs)

        if _is_self_debias(debias_model):
            debias_model._model.eval()
        else:
            debias_model.eval()
        debias_model.to(self.device)

        return debias_model
    
    def run_debias(
        self,
        path_to_crows: str,
        lang_debias: str,
        lang_eval: str,
        bias_type: str | list,
        model_class: str | None = None,
        bias_direction: str | None = None,
        projection_matrix: str | None = None,
        load_path: str | None = None,
        debias_model: torch.nn.Module | None = None,
        sample: bool = False,
    ):
        from bias_bench.util import (
            generate_experiment_id,
            _is_generative,
            _is_self_debias,
        )
        
        if model_class is None:
            model_class = self.model_class_debias
        
        if self.verbose:
            print("Running CrowS-Pairs benchmark:")
            print(f" - save_dir: {save_dir}")
            print(f" - model: {model_class}")
            print(f" - model_name_or_path: {self.model_name_or_path}")
            print(f" - bias_direction: {bias_direction}")
            print(f" - projection_matrix: {projection_matrix}")
            print(f" - load_path: {load_path}")
            print(f" - bias_type: {bias_type}")
            print(f" - sample: {sample}")
            print(f" - lang_eval: {lang_eval}")
            print(f" - lang_debias: {lang_debias}")

        if debias_model is None:
            debias_model = self._build_debias_model(
                model_class_str=model_class,
                load_path=load_path,
                bias_direction=bias_direction,
                projection_matrix=projection_matrix,
            )
        
        if hasattr(debias_model, 'to'):
            debias_model = debias_model.to(self.device)
        
        if hasattr(debias_model, 'eval'):
            debias_model.eval()
        
        runner = self.create_runner(
            batched=self.batched,
            model=debias_model,
            tokenizer=self.tokenizer,
            input_file=path_to_crows,
            lang_eval=lang_eval,
            lang_debias=lang_debias,
            bias_type=bias_type,
            is_generative=_is_generative(debias_model),
            is_self_debias=_is_self_debias(debias_model),
            sample="true" if sample else "false",
            verbose=self.verbose,
        )
        results, df_data_with_masks = runner()
        if self.save_result:
            experiment_id = generate_experiment_id(
                name="automated_test",
                model=debias_model.__class__.__name__,
                model_name_or_path=self.model_name_or_path,
                bias_type=bias_type,
                sample=sample,
                lang_eval=lang_eval,
                lang_debias=lang_debias,
            )
            
            self.save_json_output(results, "crows-debias", bias_type, lang_debias, lang_eval)

            df_data_with_masks.to_csv(self.csv_output_path(
                "crows-debias", bias_type, lang_debias, lang_eval  
            ), index=False)

        if self.verbose:
            print(f"Mejson_tric: {results}")

        return results, df_data_with_masks