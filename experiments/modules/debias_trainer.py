import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import shutil
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Literal, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
# from tqdm.auto import tqdm
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

_worker_tok = None
_worker_max_len = None

def _init_worker(tok_name: str, max_len: int):
    global _worker_tok, _worker_max_len
    _worker_tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    _worker_max_len = max_len

def _worker(text: str) -> Dict[str, Any]:
    enc = _worker_tok(
        text,
        add_special_tokens=False,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=_worker_max_len,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def _build_gender_mapping(attr_pairs):
    mapping = {}
    for male, female in attr_pairs:
        mapping[male] = female
        mapping[female] = male
    return mapping


def _build_ternary_mapping(attr_pairs):
    mapping = {}
    for triple in attr_pairs:
        w0, w1, w2 = triple
        mapping[w0] = [w1, w2]
        mapping[w1] = [w0, w2]
        mapping[w2] = [w0, w1]
    return mapping

def _process_one_sentence(
    text: str,
    tokenizer,
    max_seq_length: int,
    bias_type: Optional[str],
    word_mapping: Optional[Dict],
) -> List[Dict[str, Any]]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_seq_length,
    )
    example_batch = {"input_ids": [tokenized["input_ids"]]}

    if bias_type == "gender":
        aug_result = gender_counterfactual_augmentation(
            example_batch, word_mapping, tokenizer, max_seq_length
        )
    elif bias_type in ("race-color", "religion"):
        aug_result = ternary_counterfactual_augmentation(
            example_batch, word_mapping, tokenizer, max_seq_length
        )
    else:
        aug_result = {
            "input_ids": [tokenized["input_ids"]],
            "attention_mask": [tokenized["attention_mask"]],
            "return_special_tokens_mask": [tokenized["return_special_tokens_mask"]],
        }

    out: List[Dict[str, Any]] = []
    for i in range(len(aug_result["input_ids"])):
        out.append(
            {
                "input_ids": aug_result["input_ids"][i],
                "attention_mask": aug_result["attention_mask"][i],
            }
        )
    return out

def load_bias_attributes(json_path: Union[str, os.PathLike], bias_type: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if bias_type not in data:
        raise KeyError(f"Bias type '{bias_type}' not found in {json_path}. "
                       f"Available keys: {list(data.keys())}")
    return data[bias_type]

def _create_bias_attribute_words(attribute_file, bias_type):
    """Creates list of bias attribute words (e.g., he/she) with punctuation variants.

    Args:
        attribute_file: Path to the file containing the bias attribute words.
        bias_type: Type of bias attribute words to load. Must be one of
            ["gender", "race-color", "religion"].

    Notes:
        * We combine each bias attribute word with several punctuation marks.
          The current set of words is *not* exhaustive, however, it should
          cover most occurances.
    """
    with open(attribute_file, "r", encoding="utf-8") as f:
        bias_attribute_words = json.load(f)[bias_type]

    result = bias_attribute_words[:]
    for punctuation in [".", ",", "?", "!", ";", ":"]:
        for words in bias_attribute_words:
            augmented_words = [word + punctuation for word in words]
            result.append(augmented_words)
    return result

def gender_counterfactual_augmentation(examples, word_mapping, tokenizer, max_seq_length):
    outputs = []
    for input_ids in examples["input_ids"]:
        sentence = tokenizer.decode(input_ids)
        words = sentence.split()
        augmented = False
        new_words = words[:]
        for i, w in enumerate(words):
            if w in word_mapping:
                augmented = True
                new_words[i] = word_mapping[w]
        if augmented:
            augmented_sentence = " ".join(new_words)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    if not outputs:
        return {"input_ids": [], "attention_mask": [], "return_special_tokens_mask": []}

    return tokenizer(
        outputs,
        return_special_tokens_mask=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

def ternary_counterfactual_augmentation(examples, word_mapping, tokenizer, max_seq_length):
    outputs = []
    for input_ids in examples["input_ids"]:
        sentence = tokenizer.decode(input_ids)
        words = sentence.split()
        augmented = False
        new_words = words[:]
        for i, w in enumerate(words):
            if w in word_mapping:
                augmented = True
                new_words[i] = random.choice(word_mapping[w])
        if augmented:
            augmented_sentence = " ".join(new_words)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    if not outputs:
        return {"input_ids": [], "attention_mask": [], "return_special_tokens_mask": []}

    return tokenizer(
        outputs,
        return_special_tokens_mask=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

class CDADataset(Dataset):
    def __init__(
        self,
        raw_text_path: Union[str, os.PathLike],
        tokenizer: Any,
        max_seq_length: int,
        bias_type: Optional[str] = None,
        bias_attribute_json: Union[str, os.PathLike] = None,
        mlm_probability: float = 0.15,
        max_train_samples: Optional[int] = None,
        compute_parallel: bool = False,
        n_workers: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.max_train_samples = max_train_samples

        with open(raw_text_path, "r", encoding="utf-8") as f:
            self.raw_lines = [
                ln.strip()
                for ln in f
                if ln.strip()
            ]

        self.use_cda = bias_type is not None and bias_attribute_json is not None
        if self.use_cda:
            self.bias_type = bias_type
            self.attr_pairs = _create_bias_attribute_words(bias_attribute_json, bias_type)
            if bias_type == "gender":
                self.word_mapping = _build_gender_mapping(self.attr_pairs)
            else:
                self.word_mapping = _build_ternary_mapping(self.attr_pairs)
            self.examples = []

            if compute_parallel:
                from multiprocessing import Pool

                args = [
                    (
                        text,
                        self.tokenizer,
                        self.max_seq_length,
                        self.bias_type,
                        self.word_mapping,
                    )
                    for text in self.raw_lines
                ]

                with Pool(processes=n_workers) as pool:
                    list_of_lists = pool.starmap(_process_one_sentence, args)

                for examples in list_of_lists:
                    self.examples.extend(examples)
            else:
                for text in self.raw_lines:
                    tokenized = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        return_special_tokens_mask=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                    )
                    examples_batch = {"input_ids": [tokenized["input_ids"]]}
                    if bias_type == "gender":
                        aug_result = gender_counterfactual_augmentation(
                            examples_batch, self.word_mapping, self.tokenizer, self.max_seq_length
                        )
                    else:
                        aug_result = ternary_counterfactual_augmentation(
                            examples_batch, self.word_mapping, self.tokenizer, self.max_seq_length
                        )

                    if len(aug_result["input_ids"]) > 0:
                        for i in range(len(aug_result["input_ids"])):
                            self.examples.append(
                                {
                                    "input_ids": aug_result["input_ids"][i],
                                    "attention_mask": aug_result["attention_mask"][i],
                                }
                            )
        else:
            self.attr_pairs = None
            self.bias_type = None
            self.word_mapping = None
            self.examples = []

            if compute_parallel:
                from multiprocessing import Pool

                tokenizer_name = self.tokenizer.name_or_path
                args = [
                    (text,)
                    for text in self.raw_lines
                ]

                with Pool(
                    processes=n_workers,
                    initializer=_init_worker,
                    initargs=(tokenizer_name, self.max_seq_length),
                ) as pool:
                    self.examples = pool.map(_worker, self.raw_lines)
            else:
                for text in self.raw_lines:
                    enc = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        return_special_tokens_mask=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                    )
                    self.examples.append(
                        {
                            "input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                        }
                    )

        if self.max_train_samples is not None:
            self.examples = self.examples[: self.max_train_samples]

        del self.raw_lines

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

class BaseDebiasTrainer:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int = 128,
        seed: int = 0,
        fp16: bool = False,
        dataloader_num_workers: int = 2,
        max_train_samples: Optional[int] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.max_train_samples = max_train_samples

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )

    def _get_model_config(self, dropout_debias: bool = False):
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        if dropout_debias:
            if config.model_type in ("bert", "roberta"):
                config.hidden_dropout_prob = 0.20
                config.attention_probs_dropout_prob = 0.15
            else:
                config.hidden_dropout_prob = 0.05
                config.attention_probs_dropout_prob = 0.05
        return config

class DropoutTrainer(BaseDebiasTrainer):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int = 128,
        seed: int = 0,
        fp16: bool = False,
        dataloader_num_workers: int = 2,
        evaluator_func: Optional[callable] = None,
    ):
        super().__init__(
            model_name_or_path,
            max_seq_length=max_seq_length,
            seed=seed,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
        )
        self.evaluator_func = evaluator_func

    def train(
        self,
        *,
        train_file: Union[str, os.PathLike],
        debias_lang: str,
        output_dir: Union[str, os.PathLike] | None = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        learning_rate: float = 5e-5,
        logging_steps: int = 500,
        save_steps: Optional[int] = 500,
        overwrite_output_dir: bool = False,
        max_train_samples: Optional[int] = None,
        early_stopping_patience: int = 2,
        compute_parallel: bool = False,
        **trainer_kwargs: Any,
    ) -> str:
        tokenizer = self._get_tokenizer()
        config = self._get_model_config(dropout_debias=True)

        if output_dir is None:
            output_dir = Path(f"results/dropout/{debias_lang}/")

        model = AutoModelForMaskedLM.from_pretrained(
            self.model_name_or_path,
            config=config,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        train_dataset = CDADataset(
            raw_text_path=train_file,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            bias_type=None,
            bias_attribute_json=None,
            max_train_samples=max_train_samples,
            compute_parallel=compute_parallel,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        if overwrite_output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        callbacks = []
        if self.evaluator_func is not None:
            eval_callback = BiasEarlyStoppingCallback(
                evaluator_func=self.evaluator_func,
                target_score=50.0,
                min_threshold=48.0,
                patience=early_stopping_patience,
            )
            callbacks.append(eval_callback)
            evaluation_strategy = "steps"
        else:
            evaluation_strategy = "no"

        args = MyTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            eval_steps=logging_steps,
            seed=self.seed,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to="none",
            disable_tqdm=False,
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            debias_lang=debias_lang,
            bias_type=["gender", "race-color", "religion"],
            save_steps=save_steps,
            **trainer_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            callbacks=callbacks,
        )

        trainer.train()
        trainer.save_model()
        return str(output_dir)

class CDATrainer(BaseDebiasTrainer):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int = 128,
        seed: int = 0,
        fp16: bool = False,
        dataloader_num_workers: int = 2,
        evaluator_func: Optional[callable] = None,
    ):
        super().__init__(
            model_name_or_path,
            max_seq_length=max_seq_length,
            seed=seed,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
        )

        self.evaluator_func = evaluator_func

    def train(
        self,
        *,
        train_file: Union[str, os.PathLike],
        debias_lang: str,
        bias_attribute_json: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike] | None = None,
        bias_type: Literal["gender", "race-color", "religion"],
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        learning_rate: float = 5e-5,
        logging_steps: int = 500,
        save_steps: Optional[int] = 500,
        overwrite_output_dir: bool = False,
        max_train_samples: Optional[int] = None,
        early_stopping_patience: int = 2,
        compute_parallel: bool = False,
        **trainer_kwargs: Any,
    ) -> str:
        tokenizer = self._get_tokenizer()
        config = self._get_model_config(dropout_debias=False)

        self.bias_attribute_json = Path(bias_attribute_json)
        if not self.bias_attribute_json.is_file():
            raise FileNotFoundError(
                f"Bias attribute file not found: {self.bias_attribute_json}"
            )

        if output_dir is None:
            output_dir = Path(f"results/cda/{debias_lang}/{bias_type}")

        model = AutoModelForMaskedLM.from_pretrained(
            self.model_name_or_path,
            config=config,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        train_dataset = CDADataset(
            raw_text_path=train_file,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            bias_type=bias_type,
            bias_attribute_json=self.bias_attribute_json,
            max_train_samples=max_train_samples,
            compute_parallel=compute_parallel,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        if overwrite_output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        callbacks = []
        if self.evaluator_func is not None:
            eval_callback = BiasEarlyStoppingCallback(
                evaluator_func=self.evaluator_func,
                target_score=50.0,
                min_threshold=48.0,
                patience=early_stopping_patience,
            )
            callbacks.append(eval_callback)
            evaluation_strategy = "steps"
        else:
            evaluation_strategy = "no"

        args = MyTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            eval_steps=logging_steps,
            seed=self.seed,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to="none",
            disable_tqdm=False,
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            debias_lang=debias_lang,
            bias_type=bias_type,
            save_steps=save_steps,
            **trainer_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            callbacks=callbacks,
        )

        trainer.train()
        trainer.save_model()
        return str(output_dir)

@dataclass
class MyTrainingArguments(TrainingArguments):
    debias_lang: str = "en_US"
    bias_type: str = "gender"

class BiasEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        evaluator_func,
        target_score=50.0,
        min_threshold=48.0,
        patience=2,
    ):
        self.evaluator = evaluator_func
        self.target = target_score
        self.min_threshold = min_threshold
        self.patience = patience
        self.best_score = None
        self.wait = 0
        self.last_evaluation_step = 0
        self.steps_since_last_eval = 0
        
        self.last_bias_score = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % args.eval_steps != 0:
            return

        current_model = kwargs.get("model")
        if current_model is None:
            return

        was_training = current_model.training
        current_model.eval()

        try:
            with torch.no_grad():
                score = self.evaluator(current_model, args.bias_type, args.debias_lang)
        except Exception as e:
            print(f"Error while evaluating: {e}")
            return
        finally:
            if was_training:
                current_model.train()

        # print(f"Step {state.global_step}: BiasScore = {score:.2f}%")

        distance = abs(score - self.target)
        
        self.last_bias_score = {"bias_score": score, "new_best": False}

        if self.best_score is None or distance < self.best_score["distance"]:
            self.best_score = {
                "epoch": state.epoch,
                "step": state.global_step,
                "score": score,
                "distance": distance,
            }

            best_model_dir = os.path.join(args.output_dir, "best_model")

            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)

            os.makedirs(best_model_dir, exist_ok=True)

            current_model.save_pretrained(best_model_dir)

            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                self.tokenizer.save_pretrained(best_model_dir)

            self.wait = 0
            self.last_bias_score["new_best"] = True

            print(f"New best model: step={state.global_step}, score={score:.2f}%")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Patience limit reached ({self.wait}), stopping training...")
                control.should_training_stop = True
                return

        self.last_evaluation_step = state.global_step
        
        if score == 50.00:
            control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if self.best_score:
            print(f"Best result: Step {self.best_score['step']} with Score {self.best_score['score']:.2f}%")
            print(f"Best model: {os.path.join(args.output_dir, 'best_model')}")
            
    # def on_log(
    #     self,
    #     args,
    #     state,
    #     control,
    #     logs = None,
    #     **kwargs,
    # ):
    #     if logs is None:
    #         logs = {}
            
    #     if self.last_bias_score is not None:
    #         logs["bias_score"] = self.last_bias_score["bias_score"]
    #         log["new_best"] = self.last_bias_score["new_best"]
    #         self.last_bias_score = None