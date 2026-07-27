import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import transformers
transformers.logging.set_verbosity_error()
from functools import lru_cache
from bias_bench.model import models
from bias_bench.util import _is_generative, _is_self_debias


class CrowSPairsRunnerWrapper:
    def __init__(
        self,
        model_name_or_path: str = "bert-base-multilingual-uncased",
        model_class_base: str = "BertForMaskedLM",
        model_class_debias: str = "SentenceDebiasBertForMaskedLM",
        device: str | None = None,
    ):  
        self.model_name_or_path = model_name_or_path
        self.model_class_base = model_class_base
        self.model_class_debias = model_class_debias
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_base = None
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    def _build_base_model(self):
        self.model_base = getattr(models, self.model_class_base)(self.model_name_or_path)
        self.model_base.eval()
        
        self.model_base.to(self.device)
        
        self._is_generative = _is_generative(self.model_base)
        

    def run_plain(
        self,
        path_to_crows: str,
        lang_eval: str,
        bias_type: str,
        sample = False,
        save_dir = ".",
        save_result: bool = False,
        verbose: bool = False,
    ):
        from bias_bench.benchmark.crows import CrowSPairsRunner
        from bias_bench.util import generate_experiment_id, _is_generative
        
        if self.model_base is None:
            self._build_base_model()
        
        if verbose:
            print("Running CrowS-Pairs benchmark:")
            print(f" - save_dir: {save_dir}")
            print(f" - model_name_or_path: {self.model_name_or_path}")
            print(f" - bias_type: {bias_type}")
            print(f" - sample: {sample}")

        runner = CrowSPairsRunner(
            model=self.model_base,
            tokenizer=self.tokenizer,
            input_file=path_to_crows,
            lang_eval=lang_eval,
            bias_type=bias_type,
            is_generative=_is_generative(self.model_base),
            sample=sample,
            verbose=verbose,
        )
        results, df_data_with_masks = runner()

        if save_result:
            experiment_id = generate_experiment_id(
                name="automated_test",
                model=self.model_class_base,
                model_name_or_path=self.model_name_or_path,
                bias_type=bias_type,
                sample=sample,
                lang_eval=lang_eval,
            )

            results_dir = f"{save_dir}/results/automated_crows_test"
            os.makedirs(results_dir, exist_ok=True)
            
            with open(f"{results_dir}/{experiment_id}.json", "w") as f:
                json.dump(results, f)

            print(f"Results saved to: {results_dir}/{experiment_id}.json")

        if verbose:
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
        bias_type: str,
        model_class: str | None = None,
        bias_direction: str | None = None,
        projection_matrix: str | None = None,
        load_path: str | None = None,
        sample: bool = False,
        save_dir: str = ".",
        save_result: bool = False,
        verbose: bool = False,
    ):
        import os
        import json
        import torch
        from bias_bench.benchmark.crows import CrowSPairsRunner
        from bias_bench.util import (
            generate_experiment_id,
            _is_generative,
            _is_self_debias,
        )
        
        if model_class is None:
            model_class = self.model_class_debias
        
        if verbose:
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

        debias_model = self._build_debias_model(
            model_class_str=model_class,
            load_path=load_path,
            bias_direction=bias_direction,
            projection_matrix=projection_matrix,
        )

        tokenizer = self.tokenizer
        
        runner = CrowSPairsRunner(
            model=debias_model,
            tokenizer=tokenizer,
            input_file=path_to_crows,
            lang_eval=lang_eval,
            lang_debias=lang_debias,
            bias_type=bias_type,
            is_generative=_is_generative(debias_model),
            is_self_debias=_is_self_debias(debias_model),
            sample="true" if sample else "false",
            verbose=verbose,
        )
        results, df_data_with_masks = runner()
        if save_result:
            experiment_id = generate_experiment_id(
                name="automated_test",
                model=debias_model.__class__.__name__,
                model_name_or_path=self.model_name_or_path,
                bias_type=bias_type,
                sample=sample,
                lang_eval=lang_eval,
                lang_debias=lang_debias,
            )

            # Make sure the target directory exists
            results_dir = os.path.join(save_dir, "results", "automated_crows_debias_test")
            os.makedirs(results_dir, exist_ok=True)

            csv_path = os.path.join(
                results_dir,
                f"{experiment_id}{'_debiased' if debias_model.__class__.__name__.startswith('Debias') else ''}.csv",
            )
            df_data_with_masks.to_csv(csv_path, index=False)

            json_path = os.path.join(results_dir, f"{experiment_id}.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

            if verbose:
                print(f"Results saved to:\n  CSV: {csv_path}\n  JSON: {json_path}")

        if verbose:
            print(f"Metric: {results}")

        return results, df_data_with_masks