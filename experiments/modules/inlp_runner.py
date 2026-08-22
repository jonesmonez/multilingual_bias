import json
import nltk
import torch
import random
import sklearn
import transformers
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from bias_bench.model import models
from bias_bench.debias.inlp import debias

from experiments.modules.experiment_name  import filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InlpRunner:
    def __init__(
        self,
        model_class: str,
        model_name_or_path: str,
        save_result: bool = False,
        save_path: Path = Path("results/"),
        verbose: bool = False,
    ):
        self.model_class = model_class
        self.model_name_or_path = model_name_or_path
        self.save_result = save_result
        self.save_path = save_path
        self.verbose = verbose
        
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

        male_sentences = []
        female_sentences = []

        male_sentences_clipped = []
        female_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_male_sentences = 0
        count_female_sentences = 0
        count_neutral_sentences = 0

        for line in tqdm(lines, desc="Loading INLP data"):
            # Each line contains a paragraph of text.
            sentences = nltk.sent_tokenize(line.lower())

            for sentence in sentences:
                male_flag = False
                female_flag = False

                idx = -1
                tokens = sentence.split(" ")

                # Convert tokens to lower case.
                tokens = [token.lower() for token in tokens]

                # Skip sentences that are too short.
                if len(tokens) < 5:
                    continue

                for token in tokens:
                    # Find male definitional token.
                    if token in male_biased_token_set:
                        male_flag = True
                        idx = tokens.index(token)

                    # Find female definitional token.
                    if token in female_biased_token_set:
                        female_flag = True
                        idx = tokens.index(token)

                    # Both female and male tokens appear.
                    if male_flag and female_flag:
                        break

                # If the sentence doesn't contain male or female tokens we consider
                # it neutral.
                if (
                    not male_flag
                    and not female_flag
                    and count_neutral_sentences < n_sentences
                ):
                    # Start from the fourth token.
                    index = random.randint(4, len(tokens))
                    neutral_sentences_clipped.append(" ".join(tokens[:index]))
                    count_neutral_sentences += 1
                    continue

                # Both female and male tokens appear.
                if male_flag and female_flag:
                    continue

                if male_flag and count_male_sentences < n_sentences:
                    # Prevent duplicate sentences.
                    if sentence not in male_sentences:
                        male_sentences.append(sentence)
                        index = random.randint(idx, len(tokens))
                        male_sentences_clipped.append(" ".join(tokens[: index + 1]))
                        count_male_sentences += 1

                if female_flag and count_female_sentences < n_sentences:
                    if sentence not in female_sentences:
                        female_sentences.append(sentence)
                        index = random.randint(idx, len(tokens))
                        female_sentences_clipped.append(" ".join(tokens[: index + 1]))
                        count_female_sentences += 1

            if (
                count_male_sentences
                == count_female_sentences
                == count_neutral_sentences
                == n_sentences
            ):
                if self.verbose:
                    print("INLP dataset collected:")
                    print(f" - Num. male sentences: {count_male_sentences}")
                    print(f" - Num. female sentences: {count_female_sentences}")
                    print(f" - Num. neutral sentences: {count_neutral_sentences}")
                break

        data = {
            "male": male_sentences_clipped,
            "female": female_sentences_clipped,
            "neutral": neutral_sentences_clipped,
        }

        return data

    def _load_race_data(self):
        with open(f"{self.path_to_bias_attributes}", "r", encoding="UTF-8") as f:
            attribute_words = json.load(f)["race-color"]
        with open(f"{self.path_to_dataset}", "r", encoding="UTF-8") as f:
            lines = f.readlines()
        random.shuffle(lines)

        # Flatten the list of race words.
        race_biased_token_set = set([word for words in attribute_words for word in words])

        race_sentences = []
        race_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_race_sentences = 0
        count_neutral_sentences = 0

        

        for line in tqdm(lines, desc="Loading INLP data"):
            # Each line contains a paragraph of text.
            sentences = nltk.sent_tokenize(line.lower())

            for sentence in sentences:
                race_flag = False

                idx = -1
                tokens = sentence.split(" ")

                # Convert tokens to lower case.
                tokens = [token.lower() for token in tokens]

                # Skip sentences that are too short.
                if len(tokens) < 5:
                    continue

                for token in tokens:
                    if token in race_biased_token_set:
                        race_flag = True
                        idx = tokens.index(token)

                # If the sentence doesn't contain a racial word we consider it neutral.
                if not race_flag and count_neutral_sentences < n_sentences:
                    # Start from the fourth token.
                    index = random.randint(4, len(tokens))
                    neutral_sentences_clipped.append(" ".join(tokens[:index]))
                    count_neutral_sentences += 1
                    continue

                if race_flag and count_race_sentences < n_sentences:
                    # Prevent duplicate sentences.
                    if sentence not in race_sentences:
                        race_sentences.append(sentence)
                        index = random.randint(idx, len(tokens))
                        race_sentences_clipped.append(" ".join(tokens[: index + 1]))
                        count_race_sentences += 1

            if count_race_sentences == count_neutral_sentences == n_sentences:
                if self.verbose:
                    print("INLP dataset collected:")
                    print(f" - Num. bias sentences: {count_race_sentences}")
                    print(f" - Num. neutral sentences: {count_neutral_sentences}")
                break

        data = {"bias": race_sentences_clipped, "neutral": neutral_sentences_clipped}

        return data

    def _load_religion_data(self):
        with open(f"{self.path_to_bias_attributes}", "r", encoding="UTF-8") as f:
            attribute_words = json.load(f)["religion"]
        with open(f"{self.path_to_dataset}", "r", encoding="UTF-8") as f:
            lines = f.readlines()
        random.shuffle(lines)

        # Flatten the list of race words.
        religion_biased_token_set = set(
            [word for words in attribute_words for word in words]
        )

        religion_sentences = []
        religion_sentences_clipped = []
        neutral_sentences_clipped = []

        # We collect 10000 of each class of sentences.
        n_sentences = 10000
        count_religion_sentences = 0
        count_neutral_sentences = 0

        for line in tqdm(lines, desc="Loading INLP data"):
            # Each line contains a paragraph of text.
            sentences = nltk.sent_tokenize(line.lower())

            for sentence in sentences:
                religion_flag = False

                idx = -1
                tokens = sentence.split(" ")

                # Convert tokens to lower case.
                tokens = [token.lower() for token in tokens]

                # Skip sentences that are too short.
                if len(tokens) < 5:
                    continue

                for token in tokens:
                    if token in religion_biased_token_set:
                        religion_flag = True
                        idx = tokens.index(token)

                # If the sentence doesn't contain a religious word we consider it neutral.
                if not religion_flag and count_neutral_sentences < n_sentences:
                    index = random.randint(4, len(tokens))
                    neutral_sentences_clipped.append(" ".join(tokens[:index]))
                    count_neutral_sentences += 1
                    continue

                if religion_flag and count_religion_sentences < n_sentences:
                    # Prevent duplicate sentences.
                    if sentence not in religion_sentences:
                        religion_sentences.append(sentence)
                        index = random.randint(idx, len(tokens))
                        religion_sentences_clipped.append(" ".join(tokens[: index + 1]))
                        count_religion_sentences += 1

            if count_religion_sentences == count_neutral_sentences == n_sentences:
                if self.verbose:
                    print("INLP dataset collected:")
                    print(f" - Num. bias sentences: {count_religion_sentences}")
                    print(f" - Num. neutral sentences: {count_neutral_sentences}")
                break

        data = {"bias": religion_sentences_clipped, "neutral": neutral_sentences_clipped}

        return data


    def _extract_gender_features(
        self,
        model,
        tokenizer,
        male_sentences,
        female_sentences,
        neutral_sentences,
    ):
        """Encodes gender sentences to create a set of representations to train classifiers
        for INLP on.

        Notes:
            * Implementation taken from  https://github.com/pliang279/LM_bias.
        """
        model.to(device)

        male_features = []
        female_features = []
        neutral_features = []

        # Encode the sentences.
        with torch.no_grad():
            for sentence in tqdm(male_sentences, desc="Encoding male sentences"):
                input_ids = tokenizer(
                    sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                ).to(device)

                outputs = model(**input_ids)["last_hidden_state"]
                outputs = torch.mean(outputs, dim=1)
                outputs = outputs.squeeze().detach().cpu().numpy()

                male_features.append(outputs)

            for sentence in tqdm(female_sentences, desc="Encoding female sentences"):
                input_ids = tokenizer(
                    sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                ).to(device)

                outputs = model(**input_ids)["last_hidden_state"]
                outputs = torch.mean(outputs, dim=1)
                outputs = outputs.squeeze().detach().cpu().numpy()

                female_features.append(outputs)

            for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
                input_ids = tokenizer(
                    sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                ).to(device)

                outputs = model(**input_ids)["last_hidden_state"]
                outputs = torch.mean(outputs, dim=1)
                outputs = outputs.squeeze().detach().cpu().numpy()

                neutral_features.append(outputs)

        male_features = np.array(male_features)
        female_features = np.array(female_features)
        neutral_features = np.array(neutral_features)

        return male_features, female_features, neutral_features

    def _extract_binary_features(self, model, tokenizer, bias_sentences, neutral_sentences):
        """Encodes race/religion sentences to create a set of representations to train classifiers
        for INLP on.

        Notes:
            * Sentences are split into two classes based upon if they contain *any* race/religion bias
            attribute words.
        """
        model.to(device)

        bias_features = []
        neutral_features = []

        # Encode the sentences.
        with torch.no_grad():
            for sentence in tqdm(bias_sentences, desc="Encoding bias sentences"):
                input_ids = tokenizer(
                    sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                ).to(device)

                outputs = model(**input_ids)["last_hidden_state"]
                outputs = torch.mean(outputs, dim=1)
                outputs = outputs.squeeze().detach().cpu().numpy()

                bias_features.append(outputs)

            for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
                input_ids = tokenizer(
                    sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
                ).to(device)

                outputs = model(**input_ids)["last_hidden_state"]
                outputs = torch.mean(outputs, dim=1)
                outputs = outputs.squeeze().detach().cpu().numpy()

                neutral_features.append(outputs)

        bias_features = np.array(bias_features)
        neutral_features = np.array(neutral_features)

        return bias_features, neutral_features

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
            name = filename("inlp", self.bias_type, self.lang_debias) + ".pt"
            self.save_path.mkdir(parents=True, exist_ok=True)
            torch.save(P, self.save_path / name)
            if self.verbose:
                print(f"Saving to \"{self.save_dir}projectionmatrix.pt\"")
        
        return P