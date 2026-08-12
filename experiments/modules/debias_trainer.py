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


def _create_bias_attribute_words(attribute_file, bias_type):
    """Creates list of bias attribute words (e.g., he/she) with punctuation variants.

    Args:
        attribute_file: Path to the file containing the bias attribute words.
        bias_type: Type of bias attribute words to load. Must be one of
            ["gender", "race", "religion"].

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


def gender_counterfactual_augmentation(examples, bias_attribute_words, tokenizer, max_seq_length):
    """Applies gender counterfactual data augmentation to a batch of examples.

    Notes:
        * We apply CDA after the examples have potentially been grouped.
        * This implementation can be made more efficient by operating on
          token IDs as opposed to text. We currently decode each example
          as it is simpler.
    """
    outputs = []
    for input_ids in examples["input_ids"]:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = tokenizer.decode(input_ids)
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        augmented = False
        for position, word in enumerate(words):
            for male_word, female_word in bias_attribute_words:
                if male_word == word:
                    augmented = True
                    augmented_sentence[position] = female_word

                if female_word == word:
                    augmented = True
                    augmented_sentence[position] = male_word

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": [], "return_special_tokens_mask": []}

    return tokenizer(
        outputs,
        return_special_tokens_mask=True,
        add_special_tokens=False,  # Special tokens are already added.
        truncation=True,
        max_length=max_seq_length,
    )


def ternary_counterfactual_augmentation(examples, bias_attribute_words, tokenizer, max_seq_length):
    """Applies racial/religious counterfactual data augmentation to a batch of
    examples.

    Notes:
        * We apply CDA after the examples have potentially been grouped.
        * This implementation can be made more efficient by operating on
          token IDs as opposed to text. We currently decode each example
          as it is simpler.
    """
    outputs = []
    for input_ids in examples["input_ids"]:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = tokenizer.decode(input_ids)
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        # Sample the augmentation pairs.
        r1_augmentation_pair = random.choice([1, 2])
        r2_augmentation_pair = random.choice([0, 2])
        r3_augmentation_pair = random.choice([0, 1])

        augmented = False
        for position, word in enumerate(words):
            for augmentation_words in bias_attribute_words:
                # Implementation here.
                r1_word, r2_word, r3_word = augmentation_words

                if r1_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r1_augmentation_pair
                    ]

                if r2_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r2_augmentation_pair
                    ]

                if r3_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r3_augmentation_pair
                    ]

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": [], "return_special_tokens_mask": []}

    return tokenizer(
        outputs,
        return_special_tokens_mask=True,
        add_special_tokens=False,  # Special tokens are already added.
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

        # Apply max_train_samples after augmentation (to match run_mlm)
        self.use_cda = bias_type is not None and bias_attribute_json is not None
        if self.use_cda:
            self.bias_type = bias_type
            # Create bias attribute words with punctuation
            self.attr_pairs = _create_bias_attribute_words(bias_attribute_json, bias_type)
            # Precompute all augmented examples (tokenized) to avoid length mismatch
            self.examples = []
            for text in self.raw_lines:
                # Prepare a single-example batch for the augmentation functions
                # Tokenize without special tokens to match run_mlm's expectation
                tokenized = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_special_tokens_mask=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                # The augmentation functions expect a dict with "input_ids" list of list
                examples_batch = {"input_ids": [tokenized["input_ids"]]}
                if bias_type == "gender":
                    aug_result = gender_counterfactual_augmentation(
                        examples_batch, self.attr_pairs, self.tokenizer, self.max_seq_length
                    )
                else:  # race or religion
                    aug_result = ternary_counterfactual_augmentation(
                        examples_batch, self.attr_pairs, self.tokenizer, self.max_seq_length
                    )
                # aug_result contains tokenized dicts for augmented + original (if any)
                # Extract input_ids and attention_mask lists
                if len(aug_result["input_ids"]) > 0:
                    for i in range(len(aug_result["input_ids"])):
                        self.examples.append(
                            {
                                "input_ids": aug_result["input_ids"][i],
                                "attention_mask": aug_result["attention_mask"][i],
                            }
                        )
                # else: no augmentation -> skip (example removed)
        else:
            self.attr_pairs = None
            self.bias_type = None
            # No augmentation, tokenize raw lines
            self.examples = []
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

        # Apply max_train_samples after building examples
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
        seed: int = 0,
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