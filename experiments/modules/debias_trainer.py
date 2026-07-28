import os
import json
import shutil
import random
from pathlib import Path
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
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def load_bias_attributes(json_path: Union[str, os.PathLike], bias_type: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if bias_type not in data:
        raise KeyError(f"Bias type '{bias_type}' not found in {json_path}. "
                       f"Available keys: {list(data.keys())}")
    return data[bias_type]

def _replace_word_in_text(words: List[str], old: str, new: str) -> List[str]:
    return [new if w == old else w for w in words]

def apply_cda_to_example(
    text: str,
    bias_type: str,
    attr_pairs: List[List[str]],
) -> List[Dict[str, str]]:
    words = text.split()
    examples = []

    if bias_type == "gender":
        for fem, masc in attr_pairs:
            if fem in words:
                examples.append(
                    {
                        "original": text,
                        "female": text,
                        "male": " ".join(_replace_word_in_text(words, fem, masc)),
                    }
                )
            if masc in words:
                examples.append(
                    {
                        "original": text,
                        "female": " ".join(_replace_word_in_text(words, masc, fem)),
                        "male": text,
                    }
                )
    elif bias_type in ("race", "religion"):
        for r1, r2, r3 in attr_pairs:
            if r1 in words:
                examples.append(
                    {
                        "original": text,
                        "r1": text,
                        "r2": " ".join(_replace_word_in_text(words, r1, r2)),
                        "r3": " ".join(_replace_word_in_text(words, r1, r3)),
                    }
                )
            if r2 in words:
                examples.append(
                    {
                        "original": text,
                        "r1": " ".join(_replace_word_in_text(words, r2, r1)),
                        "r2": text,
                        "r3": " ".join(_replace_word_in_text(words, r2, r3)),
                    }
                )
            if r3 in words:
                examples.append(
                    {
                        "original": text,
                        "r1": " ".join(_replace_word_in_text(words, r3, r1)),
                        "r2": " ".join(_replace_word_in_text(words, r3, r2)),
                        "r3": text,
                    }
                )
    else:
        raise ValueError(f"Unsupported bias_type: {bias_type}")

    return examples


class CDADataset(Dataset):
    def __init__(
        self,
        raw_text_path: Union[str, os.PathLike],
        tokenizer: Any,
        max_seq_length: int,
        bias_type: Optional[str] = None,
        bias_attribute_json: Optional[Union[str, os.PathLike]] = None,
        mlm_probability: float = 0.15,
        max_train_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.max_train_samples = max_train_samples

        with open(raw_text_path, "r", encoding="utf-8") as f:
            self.raw_lines = [
                ln.strip()
                for ln in tqdm(f, desc="Loading corpus", unit="line")
                if ln.strip()
            ]
            
        if self.max_train_samples is not None:
            self.raw_lines = self.raw_lines[: self.max_train_samples]

        self.use_cda = bias_type is not None and bias_attribute_json is not None
        if self.use_cda:
            self.attr_pairs = load_bias_attributes(bias_attribute_json, bias_type)
            self.bias_type = bias_type
        else:
            self.attr_pairs = None
            self.bias_type = None

    def __len__(self) -> int:
        if not self.use_cda:
            return len(self.raw_lines)
        return len(self.raw_lines) * 6

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        line_idx = idx // 6 if self.use_cda else idx
        text = self.raw_lines[line_idx]

        if not self.use_cda:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True,
            )
        else:
            aug_examples = apply_cda_to_example(text, self.bias_type, self.attr_pairs)
            if not aug_examples:
                aug_examples = [{"original": text, "female": text, "male": text}]

            chosen = random.choice(aug_examples)
            aug_text = None
            for k, v in chosen.items():
                if k != "original":
                    aug_text = v
                    break
            if aug_text is None:
                aug_text = chosen["original"]

            enc = self.tokenizer(
                aug_text,
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True,
            )

        return enc

class BaseDebiasTrainer:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int = 128,
        seed: int = 42,
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
    def train(
        self,
        *,
        train_file: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        learning_rate: float = 5e-5,
        logging_steps: int = 500,
        save_steps: Optional[int] = None,
        overwrite_output_dir: bool = False,
        max_train_samples: Optional[int] = None,
        **trainer_kwargs: Any,
    ) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self._get_tokenizer()
        config = self._get_model_config(dropout_debias=True)

        model = AutoModelForMaskedLM.from_pretrained(
            self.model_name_or_path,
            config=config,
        )

        train_dataset = CDADataset(
            raw_text_path=train_file,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            bias_type=None,
            bias_attribute_json=None,
            max_train_samples=max_train_samples,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        if overwrite_output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            seed=self.seed,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to="none",
            disable_tqdm=False,
            **trainer_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model()
        return str(output_dir)


class CDATrainer(BaseDebiasTrainer):
    def __init__(
        self,
        model_name_or_path: str,
        bias_attribute_json: Union[str, os.PathLike],
        *,
        max_seq_length: int = 128,
        seed: int = 42,
        fp16: bool = False,
        dataloader_num_workers: int = 2,
    ):
        super().__init__(
            model_name_or_path,
            max_seq_length=max_seq_length,
            seed=seed,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
        )
        self.bias_attribute_json = Path(bias_attribute_json)
        if not self.bias_attribute_json.is_file():
            raise FileNotFoundError(
                f"Bias attribute file not found: {self.bias_attribute_json}"
            )

    def train(
        self,
        *,
        train_file: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        bias_type: Literal["gender", "race", "religion"],
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        learning_rate: float = 5e-5,
        logging_steps: int = 500,
        save_steps: Optional[int] = None,
        overwrite_output_dir: bool = False,
        max_train_samples: Optional[int] = None,
        **trainer_kwargs: Any,
    ) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self._get_tokenizer()
        config = self._get_model_config(dropout_debias=False)

        model = AutoModelForMaskedLM.from_pretrained(
            self.model_name_or_path,
            config=config,
        )

        train_dataset = CDADataset(
            raw_text_path=train_file,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            bias_type=bias_type,
            bias_attribute_json=self.bias_attribute_json,
            max_train_samples=max_train_samples,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        if overwrite_output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            seed=self.seed,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to="none",
            disable_tqdm=False,
            **trainer_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model()
        return str(output_dir)