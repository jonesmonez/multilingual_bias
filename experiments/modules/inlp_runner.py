import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import json
import nltk
import torch
import jieba
import random
import sklearn
import itertools
import numpy as np
import transformers
from zh_sentence.tokenizer import tokenize
transformers.logging.set_verbosity_error()
from pathlib import Path
from functools import partial
import multiprocessing as mp
from sklearn.svm import LinearSVC
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from bias_bench.model import models
from bias_bench.debias.inlp import debias

from concurrent.futures import ProcessPoolExecutor
from experiments.modules.experiment_name import filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _tokenize_paragraph(paragraph: str, lang: str):
    lang_map = {
        "en_US": "english",
        "de_DE": "german",
        "es_AR": "spanish",
        "ca_ES": "spanish",
        "es_ES": "spanish",
        "fr_FR": "french",
        "it_IT": "italian",
    }
    if lang == "zh_CN":
        sentence_list = tokenize(paragraph)
        return sentence_list
    else:
        nltk_lang = lang_map.get(lang, "english")
    return [
        sentence.lower()
        for sentence in nltk.sent_tokenize(paragraph, nltk_lang)
    ]

def _parallel_sent_tokenize(lines, langs, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
        
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        sentences = list(
            itertools.chain.from_iterable(
                executor.map(_tokenize_paragraph, lines, langs, chunksize=10)
            )
        )
        
    return sentences

def _encode_sentence_batched(
    sentences,
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 128,
    show_progress: bool = True,
    sent_type: str = "",
):
    sent_type = sent_type + " " if sent_type != "" else ""
    model.to(device)
    model.eval()
    all_vecs = []
    
    batch_iter = (
        tqdm(
            range(0, len(sentences), batch_size),
            desc=f"Encoding {sent_type}sentences (batched)",
            leave=True,
            disable=not show_progress,
        )
        if show_progress
        else range(0, len(sentences), batch_size)
    )
    
    for start in batch_iter:
        batch_sentences = sentences[start: start + batch_size]
        
        encoded = tokenizer(
            batch_sentences,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)["last_hidden_state"]
            mask = encoded["attention_mask"].unsqueeze(-1)
            summed = torch.sum(outputs * mask, dim=1)
            lengths = mask.sum(dim=1)
            mean_pooled = summed / lengths
            
        all_vecs.append(mean_pooled.cpu().numpy())
        
    return np.concatenate(all_vecs, axis=0)

def chinese_bias_aware_tokenize(sentence, attributes):
    for attribute in sorted(set(attributes), key=len, reverse=True):
        jieba.add_word(attribute, freq=10**8)

    return list(jieba.cut(sentence, cut_all=False))

def tokenize_for_bias(sentence, language, attributes=None):
    if language == "zh_CN":
        return chinese_bias_aware_tokenize(sentence, attributes)
    
    if language == "ca_ES":
        return re.findall(r"[\w·]+|[^\w\s·]", sentence, re.UNICODE)
    
    return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)

def _clip_at_token_index(original_sentence: str, token_index: int, tokens: list):
    current_pos = 0
    target_end = len(original_sentence)
    
    for i, token in enumerate(tokens[:token_index]):
        found_pos = original_sentence.find(token, current_pos)
        if found_pos == -1:
            return original_sentence
        current_pos = found_pos + len(token)
        if i == token_index - 1:
            target_end = current_pos
    
    return original_sentence[:target_end].strip()

class InlpRunner:
    def __init__(
        self,
        model_class: str,
        model_name_or_path: str,
        save_result: bool = False,
        save_path: Path | None = None,
        verbose: bool = False,
        compute_parallel: bool = False,
        n_workers: int | None = None,
        batch_size: int = 32,
    ):
        if save_path is None:
            save_path = Path(f"results/inlp/")
            
        self.model_class = model_class
        self.model_name_or_path = model_name_or_path
        self.save_result = save_result
        self.save_path = save_path
        self.verbose = verbose
        
        self.compute_parallel = compute_parallel
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.model = getattr(models, self.model_class)(self.model_name_or_path)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path)
        
    def setup_data(
        self,
        path_to_dataset: str,
        path_to_bias_attributes: str,
        lang_debias: str,
        bias_type: str,
        seed: int = 0,
    ):
        self.path_to_dataset = path_to_dataset
        self.path_to_bias_attributes = path_to_bias_attributes
        self.lang_debias = lang_debias
        self.bias_type = bias_type
        
        self.seed = seed
        
        self.data = self._load_inlp_data()

    def _load_inlp_data(self):
        random.seed(self.seed)

        if self.bias_type == "gender":
            data = self._load_gender_data()
        elif self.bias_type == "race-color":
            data = self._load_race_data()
        elif self.bias_type == "religion":
            data = self._load_religion_data()
        else:
            raise ValueError("Invalid bias_type, has to be one of ['gender', 'race-color', 'religion']")
        return data

    def _load_gender_data(self):
        with open(f"{self.path_to_bias_attributes}", "r", encoding="UTF-8") as f:
            attribute_words = json.load(f)["gender"]
        with open(f"{self.path_to_dataset}", "r", encoding="UTF-8") as f:
            lines = f.readlines()
        random.shuffle(lines)

        male_biased_token_set = set([words[0] for words in attribute_words])
        female_biased_token_set = set([words[1] for words in attribute_words])
        
        if self.compute_parallel:
            sentences = _parallel_sent_tokenize(lines, [self.lang_debias] * len(lines), n_workers=self.n_workers)
        else:
            sentences = []
            for line in tqdm(lines, desc="Loading INLP Data", leave=False):
                sentences.extend(_tokenize_paragraph(line, self.lang_debias))

        male_sentences = []
        female_sentences = []
        neutral_sentences = []

        male_sentences_clipped = []
        female_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_male = count_female = count_neutral = 0

        for sentence in sentences:
            # tokens = sentence.split(" ")
            tokens = tokenize_for_bias(
                sentence,
                self.lang_debias,
                male_biased_token_set | female_biased_token_set
            )
            
            male_flag = any(tok in male_biased_token_set for tok in tokens)
            female_flag = any(tok in female_biased_token_set for tok in tokens)
            
            if(not male_flag and not female_flag and count_neutral < n_sentences):
                if len(tokens) < 4:
                    index = len(tokens)
                else:
                    index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(_clip_at_token_index(sentence, index, tokens))
                count_neutral += 1
                continue
            
            if male_flag and count_male < n_sentences:
                if sentence not in male_sentences:
                    male_sentences.append(sentence)
                    idx = next(i for i, t in enumerate(tokens) if t in male_biased_token_set)
                    index = random.randint(idx, len(tokens))
                    clipped_sentence = _clip_at_token_index(sentence, index + 1, tokens)
                    male_sentences_clipped.append(clipped_sentence)
                    count_male += 1
                    
            if female_flag and count_female < n_sentences:
                if sentence not in female_sentences:
                    female_sentences.append(sentence)
                    idx = next(i for i, t in enumerate(tokens) if t in female_biased_token_set)
                    index = random.randint(idx, len(tokens))
                    clipped_sentence = _clip_at_token_index(sentence, index + 1, tokens)
                    female_sentences_clipped.append(clipped_sentence)
                    count_female += 1
                
            if (count_male == count_female == count_neutral == n_sentences):
                break
            
        if self.verbose:
            print("INLP dataset collected:")
            print(f" - Num. male sentences: {count_male}")
            print(f" - Num. female sentences: {count_female}")
            print(f" - Num. neutral sentences: {count_neutral}")
            
        return {
            "male": male_sentences_clipped,
            "female": female_sentences_clipped,
            "neutral": neutral_sentences_clipped,
        }

    def _load_race_data(self):
        with open(f"{self.path_to_bias_attributes}", "r", encoding="UTF-8") as f:
            attribute_words = json.load(f)["race-color"]
        with open(f"{self.path_to_dataset}", "r", encoding="UTF-8") as f:
            lines = f.readlines()
        random.shuffle(lines)

        race_biased_token_set = set(
            [word for words in attribute_words for word in words]
        )

        race_sentences = []
        race_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_race = 0
        count_neutral = 0

        if self.compute_parallel:
            sentences = _parallel_sent_tokenize(lines, [self.lang_debias] * len(lines), n_workers=self.n_workers)
        else:
            sentences = []
            for line in tqdm(lines, desc="Loading INLP data", leave=False):
                sentences.extend(_tokenize_paragraph(line, self.lang_debias))

        for sentence in sentences:
            # tokens = sentence.split(" ")
            tokens = tokenize_for_bias(
                sentence,
                self.lang_debias,
                race_biased_token_set
            )

            race_flag = any(tok in race_biased_token_set for tok in tokens)
            
            if (not race_flag and count_neutral < n_sentences):
                if len(tokens) < 4:
                    index = len(tokens)
                else:
                    index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(_clip_at_token_index(sentence, index, tokens))
                count_neutral += 1
                continue
            
            if race_flag and count_race < n_sentences:
                if sentence not in race_sentences:
                    race_sentences.append(sentence)
                    idx = next(i for i, t in enumerate(tokens) if t in race_biased_token_set)
                    index = random.randint(idx, len(tokens))
                    race_sentences_clipped.append(_clip_at_token_index(sentence, index + 1, tokens))
                    count_race += 1
                    
            if count_race == count_neutral == n_sentences:
                break

        if self.verbose:
            print("INLP dataset collected:")
            print(f" - Num. bias sentences: {count_race}")
            print(f" - Num. neutral sentences: {count_neutral}")

        return {"bias": race_sentences_clipped, "neutral": neutral_sentences_clipped}

    def _load_religion_data(self):
        with open(f"{self.path_to_bias_attributes}", "r", encoding="UTF-8") as f:
            attribute_words = json.load(f)["religion"]
        with open(f"{self.path_to_dataset}", "r", encoding="UTF-8") as f:
            lines = f.readlines()
        random.shuffle(lines)

        religion_biased_token_set = set(
            [word for words in attribute_words for word in words]
        )

        religion_sentences = []
        religion_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_religion = 0
        count_neutral = 0

        if self.compute_parallel:
            sentences = _parallel_sent_tokenize(lines, [self.lang_debias] * len(lines), n_workers=self.n_workers)
        else:
            sentences = []
            for line in tqdm(lines, desc="Loading INLP data", leave=False):
                sentences.extend(_tokenize_paragraph(line, self.lang_debias))

        for sentence in sentences:
            # tokens = sentence.split(" ")
            tokens = tokenize_for_bias(
                sentence,
                self.lang_debias,
                religion_biased_token_set
            )
            
            religion_flag = any(tok in religion_biased_token_set for tok in tokens)
            
            if (not religion_flag and count_neutral < n_sentences):
                if len(tokens) < 4:
                    index = len(tokens)
                else:
                    index = random.randint(4, len(tokens))
                neutral_sentences_clipped.append(_clip_at_token_index(sentence, index, tokens))
                count_neutral += 1
                continue

            if religion_flag and count_religion < n_sentences:
                if sentence not in religion_sentences:
                    religion_sentences.append(sentence)
                    idx = next(i for i, t in enumerate(tokens) if t in religion_biased_token_set)
                    index = random.randint(idx, len(tokens))
                    religion_sentences_clipped.append(_clip_at_token_index(sentence, index + 1, tokens))
                    count_religion += 1

            if count_religion == count_neutral == n_sentences:
                break
        
        if self.verbose:
            print("INLP dataset collected:")
            print(f" - Num. bias sentences: {count_religion}")
            print(f" - Num. neutral sentences: {count_neutral}")

        return {"bias": religion_sentences_clipped, "neutral": neutral_sentences_clipped}

    def _extract_gender_features(
        self,
        model,
        tokenizer,
        male_sentences,
        female_sentences,
        neutral_sentences,
    ):
        if self.compute_parallel:
            male_features = _encode_sentence_batched(
                male_sentences,
                tokenizer,
                model,
                device,
                batch_size=self.batch_size,
                show_progress=True,
                sent_type = "male",
            )
            female_features = _encode_sentence_batched(
                female_sentences,
                tokenizer,
                model,
                device,
                batch_size=self.batch_size,
                show_progress=True,
                sent_type = "female",
            )
            neutral_features = _encode_sentence_batched(
                neutral_sentences,
                tokenizer,
                model,
                device,
                batch_size=self.batch_size,
                show_progress=True,
                sent_type = "neutral",
            )
            return male_features, female_features, neutral_features
        else:
            model.to(device)
            model.eval()

            male_features = []
            female_features = []
            neutral_features = []

            with torch.no_grad():
                for sentence in tqdm(male_sentences, desc="Encoding male sentences", leave=True):
                    input_ids = tokenizer(
                        sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    outputs = model(**input_ids)["last_hidden_state"]
                    outputs = torch.mean(outputs, dim=1).squeeze().detach().cpu().numpy()
                    male_features.append(outputs)

                for sentence in tqdm(female_sentences, desc="Encoding female sentences", leave=True):
                    input_ids = tokenizer(
                        sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    outputs = model(**input_ids)["last_hidden_state"]
                    outputs = torch.mean(outputs, dim=1).squeeze().detach().cpu().numpy()
                    female_features.append(outputs)

                for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences", leave=True):
                    input_ids = tokenizer(
                        sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    outputs = model(**input_ids)["last_hidden_state"]
                    outputs = torch.mean(outputs, dim=1).squeeze().detach().cpu().numpy()
                    neutral_features.append(outputs)

            return np.array(male_features), np.array(female_features), np.array(neutral_features)

    def _extract_binary_features(self, model, tokenizer, bias_sentences, neutral_sentences):
        if self.compute_parallel:
            bias_features = _encode_sentence_batched(
                bias_sentences,
                tokenizer,
                model,
                device,
                batch_size=self.batch_size,
                show_progress=True,
                sent_type = "bias",
            )
            neutral_features = _encode_sentence_batched(
                neutral_sentences,
                tokenizer,
                model,
                device,
                batch_size=self.batch_size,
                show_progress=True,
                sent_type = "neutral",
            )
            return bias_features, neutral_features
        else:
            model.to(device)
            model.eval()

            bias_features = []
            neutral_features = []

            with torch.no_grad():
                for sentence in tqdm(bias_sentences, desc="Encoding bias sentences", leave=True):
                    input_ids = tokenizer(
                        sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    outputs = model(**input_ids)["last_hidden_state"]
                    outputs = torch.mean(outputs, dim=1).squeeze().detach().cpu().numpy()
                    bias_features.append(outputs)

                for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences", leave=True):
                    input_ids = tokenizer(
                        sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    outputs = model(**input_ids)["last_hidden_state"]
                    outputs = torch.mean(outputs, dim=1).squeeze().detach().cpu().numpy()
                    neutral_features.append(outputs)

            return np.array(bias_features), np.array(neutral_features)

    def _split_gender_dataset(self, male_feat, female_feat, neut_feat):
        np.random.seed(self.seed)

        X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)

        y_male = np.ones(male_feat.shape[0], dtype=int)
        y_female = np.zeros(female_feat.shape[0], dtype=int)
        y_neutral = -np.ones(neut_feat.shape[0], dtype=int)

        y = np.concatenate((y_male, y_female, y_neutral))

        X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
            X_train_dev, y_train_dev, test_size=0.3, random_state=0
        )

        return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

    def _split_binary_dataset(self, bias_feat, neut_feat):
        np.random.seed(self.seed)

        X = np.concatenate((bias_feat, neut_feat), axis=0)

        y_bias = np.ones(bias_feat.shape[0], dtype=int)
        y_neutral = np.zeros(neut_feat.shape[0], dtype=int)

        y = np.concatenate((y_bias, y_neutral))

        X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
            X_train_dev, y_train_dev, test_size=0.3, random_state=0
        )

        return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

    def _apply_nullspace_projection(
        self, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=80
    ):
        classifier_parameters = {
            "fit_intercept": False,
            "class_weight": None,
            "dual": False,
            "random_state": 0,
        }

        P, rowspace_projs, Ws = debias.get_debiasing_projection(
            classifier_class=LinearSVC,
            cls_params=classifier_parameters,
            num_classifiers=n_classifiers,
            input_dim=768,
            is_autoregressive=True,
            min_accuracy=0,
            X_train=X_train,
            Y_train=Y_train,
            X_dev=X_dev,
            Y_dev=Y_dev,
            Y_train_main=None,
            Y_dev_main=None,
            by_class=False,
            dropout_rate=0,
        )

        return P, rowspace_projs, Ws

    def compute_projection_matrix(self, n_classifiers=80):
        if self.bias_type == "gender":
            male_sentences = self.data["male"]
            female_sentences = self.data["female"]
            neutral_sentences = self.data["neutral"]

            male_features, female_features, neutral_features = self._extract_gender_features(
                self.model, self.tokenizer, male_sentences, female_sentences, neutral_sentences
            )

            X_train, X_dev, X_test, Y_train, Y_dev, Y_test = self._split_gender_dataset(
                male_features, female_features, neutral_features
            )

        else:
            bias_sentences = self.data["bias"]
            neutral_sentences = self.data["neutral"]

            bias_features, neutral_features = self._extract_binary_features(
                self.model, self.tokenizer, bias_sentences, neutral_sentences
            )

            X_train, X_dev, X_test, Y_train, Y_dev, Y_test = self._split_binary_dataset(
                bias_features, neutral_features
            )

        if self.verbose:
            print("Dataset split sizes:")
            print(
                f"Train size: {X_train.shape[0]}; Dev size: {X_dev.shape[0]}; Test size: {X_test.shape[0]}"
            )

        P, rowspace_projs, Ws = self._apply_nullspace_projection(
            X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=n_classifiers
        )

        P = torch.tensor(P, dtype=torch.float32)

        if self.save_result:
            path = self.save_path / self.lang_debias
            name = f"{self.bias_type}.pt"
            path.mkdir(parents=True, exist_ok=True)
            torch.save(P, path / name)
            if self.verbose:
                print(f"Saving to \"{path}/{path}\"")
        
        return P