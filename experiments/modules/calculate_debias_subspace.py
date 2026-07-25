import os
import json
import nltk
import torch
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
# try:
#     from tqdm.notebook import tqdm
# except ImportError:
#     from tqdm import tqdm
from tqdm import tqdm
from sklearn.decomposition import PCA

from bias_bench.model import models

class SubspaceCalculator:
    def __init__(
        self,
        model_class: str,
        model_name_or_path: str,
        batch_size: int,
        save_result: bool,
        verbose: bool,
    ):
        self.model_class = model_class
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        
        self.model = getattr(models, self.model_class)(self.model_name_or_path)
        self.model.eval()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.save_result = save_result
        self.verbose = verbose

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for encoding!")
            self.model = torch.nn.DataParallel(self.model)
            
    def setup_data(
        self,
        path_to_bias_attributes,
        lang_debias,
        path_to_dataset,
        n_max_sent = 250000,
    ):
        self.data = _GenericDataset(
            path_to_bias_attributes=path_to_bias_attributes,
            lang_debias=lang_debias,
            path_to_text_corpus=path_to_dataset
        )
        self.data.load_examples(n_max_sent)
        
    def compute_gender_subspace(self):
        try:
            self.data
        except NameError:
            print("Run .setup_data() first")
            return
        
        gender_data = self.data.collect_counterfactual_sents("gender")
        # self.n_batches = len(gender_data)
        self.n_batches = (len(gender_data) + self.batch_size - 1) // self.batch_size
        
        self.all_embeddings_male = []
        self.all_embeddings_female = []
        
        for i in tqdm(range(self.n_batches), desc="Encoding gender examples"):
            offset = self.batch_size * i

            inputs_male = self.tokenizer(
                [example["male_example"] for example in gender_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )

            inputs_female = self.tokenizer(
                [example["female_example"] for example in gender_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )

            male_input_ids = inputs_male["input_ids"].to(self.device)
            female_input_ids = inputs_female["input_ids"].to(self.device)

            male_attention_mask = inputs_male["attention_mask"].to(self.device)
            female_attention_mask = inputs_female["attention_mask"].to(self.device)

            with torch.no_grad():
                # Compute average representation from last layer.
                # embedding_male.shape == (batch_size, 128, 768).
                embedding_male = self.model(
                    input_ids=male_input_ids, attention_mask=male_attention_mask
                )["last_hidden_state"]
                embedding_male *= male_attention_mask.unsqueeze(-1)
                embedding_male = embedding_male.sum(dim=1)
                embedding_male /= male_attention_mask.sum(dim=1, keepdims=True)

                embedding_female = self.model(
                    input_ids=female_input_ids, attention_mask=female_attention_mask
                )["last_hidden_state"]
                embedding_female *= female_attention_mask.unsqueeze(-1)
                embedding_female = embedding_female.sum(dim=1)
                embedding_female /= female_attention_mask.sum(dim=1, keepdims=True)

            embedding_male /= torch.norm(embedding_male, dim=-1, keepdim=True)
            embedding_female /= torch.norm(embedding_female, dim=-1, keepdim=True)

            self.all_embeddings_male.append(embedding_male.cpu().numpy())
            self.all_embeddings_female.append(embedding_female.cpu().numpy())

        self.all_embeddings_male = np.concatenate(self.all_embeddings_male, axis=0)
        self.all_embeddings_female = np.concatenate(self.all_embeddings_female, axis=0)

class SentenceDebiasWrapper(SubspaceCalculator):
    def __init__(
        self,
        model_class: str = "BertModel",
        model_name_or_path: str = "bert-base-multilingual-uncased",
        batch_size: int = 32,
        save_result: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            model_class,
            model_name_or_path,
            batch_size,
            save_result,
            verbose,
        )
        
    def compute_gender_subspace(self, save_path: str | None = None):
        """Returns race subspace components for SentenceDebias.

        Implementation based upon: https://github.com/pliang279/sent_debias.
        """
        if self.save_result and save_path is None:
            raise ValueError("'save_path' has to be defined if 'save_results' is True")
        elif save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
                
        super().compute_gender_subspace()

        means = (self.all_embeddings_male + self.all_embeddings_female) / 2.0
        
        all_embeddings_male_mean = self.all_embeddings_male.copy()
        all_embeddings_female_mean = self.all_embeddings_female.copy()
        all_embeddings_male_mean -= means
        all_embeddings_female_mean -= means

        all_embeddings = np.concatenate(
            [all_embeddings_male_mean, all_embeddings_female_mean], axis=0
        )

        pca = PCA(n_components=1)
        pca.fit(all_embeddings)

        bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        
        if self.save_result:
            os.makedirs(f"{save_path}results/sentence/", exist_ok=True)
            torch.save(bias_direction, f"{save_path}results/sentence/gender_subspace.pt")

        return bias_direction

    def compute_racecolor_subspace(self, save_path: str | None = None):
        """Returns race subspace components for SentenceDebias.

        Implementation based upon: https://github.com/pliang279/sent_debias.
        """
        try:
            self.data
        except NameError:
            print("Run .setup_data() first")
            return
        
        race_color_data = self.data.collect_counterfactual_sents("race-color")
        self.n_batches = (len(race_color_data) + self.batch_size - 1) // self.batch_size
        
        if self.save_result and save_path is None:
            raise ValueError("'save_path' has to be defined if 'save_results' is True")
        elif save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
                
        all_embeddings_r1 = []
        all_embeddings_r2 = []
        all_embeddings_r3 = []

        for i in tqdm(range(self.n_batches), desc="Encoding race examples"):
            offset = self.batch_size * i

            inputs_r1 = self.tokenizer(
                [example["r1_example"] for example in race_color_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            inputs_r2 = self.tokenizer(
                [example["r2_example"] for example in race_color_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            inputs_r3 = self.tokenizer(
                [example["r3_example"] for example in race_color_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            r1_input_ids = inputs_r1["input_ids"].to(self.device)
            r1_attention_mask = inputs_r1["attention_mask"].to(self.device)

            r2_input_ids = inputs_r2["input_ids"].to(self.device)
            r2_attention_mask = inputs_r2["attention_mask"].to(self.device)

            r3_input_ids = inputs_r3["input_ids"].to(self.device)
            r3_attention_mask = inputs_r3["attention_mask"].to(self.device)

            with torch.no_grad():
                embedding_r1 = self.model(
                    input_ids=r1_input_ids, attention_mask=r1_attention_mask
                )["last_hidden_state"]
                embedding_r1 *= r1_attention_mask.unsqueeze(-1)
                embedding_r1 = embedding_r1.sum(dim=1)
                embedding_r1 /= r1_attention_mask.sum(dim=1, keepdims=True)

                embedding_r2 = self.model(
                    input_ids=r2_input_ids, attention_mask=r2_attention_mask
                )["last_hidden_state"]
                embedding_r2 *= r2_attention_mask.unsqueeze(-1)
                embedding_r2 = embedding_r2.sum(dim=1)
                embedding_r2 /= r2_attention_mask.sum(dim=1, keepdims=True)

                embedding_r3 = self.model(
                    input_ids=r3_input_ids, attention_mask=r3_attention_mask
                )["last_hidden_state"]
                embedding_r3 *= r3_attention_mask.unsqueeze(-1)
                embedding_r3 = embedding_r3.sum(dim=1)
                embedding_r3 /= r3_attention_mask.sum(dim=1, keepdims=True)

            embedding_r1 /= torch.norm(embedding_r1, dim=-1, keepdim=True)
            embedding_r2 /= torch.norm(embedding_r2, dim=-1, keepdim=True)
            embedding_r3 /= torch.norm(embedding_r3, dim=-1, keepdim=True)

            all_embeddings_r1.append(embedding_r1.cpu().numpy())
            all_embeddings_r2.append(embedding_r2.cpu().numpy())
            all_embeddings_r3.append(embedding_r3.cpu().numpy())

        all_embeddings_r1 = np.concatenate(all_embeddings_r1, axis=0)
        all_embeddings_r2 = np.concatenate(all_embeddings_r2, axis=0)
        all_embeddings_r3 = np.concatenate(all_embeddings_r3, axis=0)

        means = (all_embeddings_r1 + all_embeddings_r2 + all_embeddings_r3) / 3.0
        all_embeddings_r1 -= means
        all_embeddings_r2 -= means
        all_embeddings_r3 -= means

        all_embeddings = np.concatenate(
            [all_embeddings_r1, all_embeddings_r2, all_embeddings_r3], axis=0
        )

        pca = PCA(n_components=1)
        pca.fit(all_embeddings)

        bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)

        if self.save_result:
            os.makedirs(f"{save_path}results/sentence/", exist_ok=True)
            torch.save(bias_direction, f"{save_path}results/sentence/racecolor_subspace.pt")

        return bias_direction

    def compute_religion_subspace(self, save_path: str | None = None):
        """Returns religion subspace components for SentenceDebias.

        Implementation based upon: https://github.com/pliang279/sent_debias.
        """
        try:
            self.data
        except NameError:
            print("Run .setup_data() first")
            return
        
        religion_data = self.data.collect_counterfactual_sents("religion")
        self.n_batches = (len(religion_data) + self.batch_size - 1) // self.batch_size
        
        if self.save_result and save_path is None:
            raise ValueError("'save_path' has to be defined if 'save_results' is True")
        elif save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
                
        all_embeddings_r1 = []
        all_embeddings_r2 = []
        all_embeddings_r3 = []

        for i in tqdm(range(self.n_batches), desc="Encoding religion examples"):
            offset = self.batch_size * i

            inputs_r1 = self.tokenizer(
                [example["r1_example"] for example in religion_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            inputs_r2 = self.tokenizer(
                [example["r2_example"] for example in religion_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            inputs_r3 = self.tokenizer(
                [example["r3_example"] for example in religion_data[offset : offset + self.batch_size]],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ).to(self.device)

            r1_input_ids = inputs_r1["input_ids"].to(self.device)
            r1_attention_mask = inputs_r1["attention_mask"].to(self.device)

            r2_input_ids = inputs_r2["input_ids"].to(self.device)
            r2_attention_mask = inputs_r2["attention_mask"].to(self.device)

            r3_input_ids = inputs_r3["input_ids"].to(self.device)
            r3_attention_mask = inputs_r3["attention_mask"].to(self.device)

            with torch.no_grad():
                embedding_r1 = self.model(
                    input_ids=r1_input_ids, attention_mask=r1_attention_mask
                )["last_hidden_state"]
                embedding_r1 *= r1_attention_mask.unsqueeze(-1)
                embedding_r1 = embedding_r1.sum(dim=1)
                embedding_r1 /= r1_attention_mask.sum(dim=1, keepdims=True)

                embedding_r2 = self.model(
                    input_ids=r2_input_ids, attention_mask=r2_attention_mask
                )["last_hidden_state"]
                embedding_r2 *= r2_attention_mask.unsqueeze(-1)
                embedding_r2 = embedding_r2.sum(dim=1)
                embedding_r2 /= r2_attention_mask.sum(dim=1, keepdims=True)

                embedding_r3 = self.model(
                    input_ids=r3_input_ids, attention_mask=r3_attention_mask
                )["last_hidden_state"]
                embedding_r3 *= r3_attention_mask.unsqueeze(-1)
                embedding_r3 = embedding_r3.sum(dim=1)
                embedding_r3 /= r3_attention_mask.sum(dim=1, keepdims=True)

            embedding_r1 /= torch.norm(embedding_r1, dim=-1, keepdim=True)
            embedding_r2 /= torch.norm(embedding_r2, dim=-1, keepdim=True)
            embedding_r3 /= torch.norm(embedding_r3, dim=-1, keepdim=True)

            all_embeddings_r1.append(embedding_r1.cpu().numpy())
            all_embeddings_r2.append(embedding_r2.cpu().numpy())
            all_embeddings_r3.append(embedding_r3.cpu().numpy())

        # all_embeddings_r1.shape == (num_examples, dim).
        all_embeddings_r1 = np.concatenate(all_embeddings_r1, axis=0)
        all_embeddings_r2 = np.concatenate(all_embeddings_r2, axis=0)
        all_embeddings_r3 = np.concatenate(all_embeddings_r3, axis=0)

        means = (all_embeddings_r1 + all_embeddings_r2 + all_embeddings_r3) / 3.0
        all_embeddings_r1 -= means
        all_embeddings_r2 -= means
        all_embeddings_r3 -= means

        all_embeddings = np.concatenate(
            [all_embeddings_r1, all_embeddings_r2, all_embeddings_r3], axis=0
        )

        pca = PCA(n_components=1)
        pca.fit(all_embeddings)

        # We use only the first PCA component for debiasing.
        bias_direction = torch.tensor(pca.components_[0], dtype=torch.float32)

        if self.save_result:
            os.makedirs(f"{save_path}results/sentence/", exist_ok=True)
            torch.save(bias_direction, f"{save_path}results/sentence/religion_subspace.pt")

        return bias_direction

class DensrayDebiasWrapper(SubspaceCalculator):
    def __init__(
        self,
        model_class: str = "BertModel",
        model_name_or_path: str = "bert-base-multilingual-uncased",
        batch_size: int = 32,
        save_result: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            model_class,
            model_name_or_path,
            batch_size,
            save_result,
            verbose,
        )

    def compute_gender_subspace(self, save_path: str | None = None):
        if self.save_result and save_path is None:
            raise ValueError("'save_path' has to be defined if 'save_results' is True")
        elif save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
            
        super().compute_gender_subspace()
        
        print('Computing the bias dimensions')
        densray = DensrayDebiasWrapper.DensRay(torch.from_numpy(self.all_embeddings_male),torch.from_numpy(self.all_embeddings_female))
        densray.fit()
        
        result = {
            "eigenvecs": densray.eigenvecs,
            "mean": densray.mean,
            "std": densray.std
        }

        if self.save_result:
            os.makedirs(f"{save_path}results/densray", exist_ok=True)
            torch.save(result, f"{save_path}results/densray/gender_subspace.pt")
            
        return result

    class DensRay:
        def __init__(self, Lemb, Remb):
            self.lemb = Lemb
            self.remb = Remb

        def fit(self, weights=None, normalize_D=True):
            """Fit DensRay
            Args:
                weights: only for binary model; how to weight the two
                    summands; if none
                    
                    
                    : apply dynamic weighting. Example input: [1.0, 1.0]
                normalize_D: bool whether to normalize the difference vectors with l2 norm
            """
            #self.computeA_binary_part1(normalize_D=normalize_D)
            print(type(self.lemb))
            self.A_equal = self.opsum(self.lemb) + self.opsum(self.remb)
            self.A_unequal = self.opsum(self.lemb, self.remb) + self.opsum(self.remb, self.lemb)
            self.computeA_binary_part2(weights=weights)
            self.compute_trafo()
            self.compute_mean_var()

        @staticmethod
        def opsum(a, b=None):
            if b is None: b = a
            out = -torch.ger(a.sum(dim=0), b.sum(dim=0))
            out = out + out.T
            out += b.shape[0] * torch.mm(a.T,a)
            out += a.shape[0] * torch.mm(b.T,b)
            return out

        @staticmethod
        def outer_product_sub_binary(v, M, normD):
            """Helper function to compute the sum of outer products

            While it is not very readable, it is more efficient than
            a brute force implementation.
            """
            d = v.unsqueeze(0) - M
            if normD:
                norm = d.norm(dim=1)
                norm[norm == 0] = 1
                d = d / (norm.unsqueeze(0).T)
            return torch.mm(d.T, d)
        
        def computeA_binary_part1(self, normalize_D=False):
            """First part of computing the matrix A.
            Args:
                normalize_D: bool whether to normalize the difference vectors with l2 norm.
            """
            dim = self.lemb.shape[1]
            self.A_equal = torch.zeros((dim, dim)).to(device)
            self.A_unequal = torch.zeros((dim, dim)).to(device)
            for ipos in tqdm.trange(self.lemb.shape[0]):
                v = self.lemb[ipos]
                self.A_equal += self.outer_product_sub_binary(v, self.lemb, normalize_D)
                self.A_unequal += self.outer_product_sub_binary(v, self.remb, normalize_D)
            for ineg in tqdm.trange(self.remb.shape[0]):
                v = self.remb[ineg]
                self.A_equal += self.outer_product_sub_binary(v, self.remb, normalize_D)
                self.A_unequal += self.outer_product_sub_binary(v, self.lemb, normalize_D)

        def computeA_binary_part2(self, weights=None):
            """Second part of computing the matrix A.
            Args:
                weights: only for binary model; how to weight the two 
                    summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
            """
            if weights is None:
                weights = [1 / (2 * self.lemb.shape[0] * self.remb.shape[0]), 1 /
                        (self.lemb.shape[0]**2 + self.remb.shape[0]**2)]
            # normalize matrices for numerical reasons
            # note that this does not change the eigenvectors
            n1 = self.A_unequal.max()
            n2 = self.A_equal.max()
            weights = [weights[0] / max(n1, n2), weights[1] / max(n1, n2)]
            self.A = weights[0] * self.A_unequal - weights[1] * self.A_equal

        def compute_trafo(self):
            """Given A, this function computes the actual Transformation.
            It essentially just does an eigenvector decomposition.
            """
            eigvals, eigvecs = self.A.symeig(eigenvectors=True)
            # need to sort the eigenvalues
            idx = eigvals.argsort(descending=True)
            eigvals, self.eigvecs = eigvals[idx], eigvecs[:, idx]
        
        def compute_mean_var(self):
            first_dim = torch.mm(torch.cat((self.lemb, self.remb)), self.eigvecs)[:, 0]
            self.mean = first_dim.mean()
            self.std = first_dim.var().sqrt()

class _SentenceDebiasDataset:
    def _gender_augment_func(self, text, examples, attribute_words):
        words = text.split(" ")

        for i, (female_word, male_word) in enumerate(attribute_words):
            if female_word in words:
                female_example = text
                male_example = self._replace_word_in_text(female_word, male_word, words)
                examples.append(
                    {"female_example": female_example, "male_example": male_example}
                )

            if male_word in words:
                female_example = self._replace_word_in_text(male_word, female_word, words)
                male_example = text
                examples.append(
                    {"female_example": female_example, "male_example": male_example}
                )

        return examples

    def _racecolor_augment_func(self, text, examples, attribute_words):
        words = text.split(" ")

        for i, (r1_word, r2_word, r3_word) in enumerate(attribute_words):
            if r1_word in words:
                r1_example = text
                r2_example = self._replace_word_in_text(r1_word, r2_word, words)
                r3_example = self._replace_word_in_text(r1_word, r3_word, words)

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

            if r2_word in words:
                r1_example = self._replace_word_in_text(r2_word, r1_word, words)
                r2_example = text
                r3_example = self._replace_word_in_text(r2_word, r3_word, words)

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

            if r3_word in words:
                r1_example = self._replace_word_in_text(r3_word, r1_word, words)
                r2_example = self._replace_word_in_text(r3_word, r2_word, words)
                r3_example = text

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

        return examples

    def _religion_augment_func(self, text, examples, attribute_words):
        words = text.split(" ")

        for i, (r1_word, r2_word, r3_word) in enumerate(attribute_words):
            if r1_word in words:
                r1_example = text
                r2_example = self._replace_word_in_text(r1_word, r2_word, words)
                r3_example = self._replace_word_in_text(r1_word, r3_word, words)

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

            if r2_word in words:
                r1_example = self._replace_word_in_text(r2_word, r1_word, words)
                r2_example = text
                r3_example = self._replace_word_in_text(r2_word, r3_word, words)

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

            if r3_word in words:
                r1_example = self._replace_word_in_text(r3_word, r1_word, words)
                r2_example = self._replace_word_in_text(r3_word, r2_word, words)
                r3_example = text

                examples.append(
                    {
                        "r1_example": r1_example,
                        "r2_example": r2_example,
                        "r3_example": r3_example,
                    }
                )

        return examples

    def _replace_word_in_text(self, word_to_replace, new_word, words):
        return " ".join([new_word if word == word_to_replace else word for word in words])


    _bias_type_to_func = {
        "gender": _gender_augment_func,
        "race-color": _racecolor_augment_func,
        "religion": _religion_augment_func,
    }

    def __init__(self, path_to_bias_attributes, lang_debias):
        self._path_to_bias_attributes = path_to_bias_attributes
        self._lang_debias=lang_debias
        
    def load_attributes(self, bias_type):
        with open(self._path_to_bias_attributes, "r") as f:
            self._attribute_words = json.load(f)[self._bias_type]
        
    def load_examples(self):
        raise NotImplementedError("load_examples method not implemented.")

class _GenericDataset(_SentenceDebiasDataset):
    def __init__(
        self,
        path_to_bias_attributes,
        lang_debias,
        path_to_text_corpus
    ):
        super().__init__(path_to_bias_attributes, lang_debias)
        self._path_corpus = path_to_text_corpus
        
    def load_examples(self, n_max_sent):
        with open(self._path_corpus, "r") as f:
            lines = f.readlines()

        self.tokenized_data = []
        for line in tqdm(lines[:n_max_sent], desc=f"Sentence tokenizing {self._lang_debias}", leave=False):
            line = line.lower()
            self.tokenized_data.extend(nltk.sent_tokenize(line))
        

    def collect_counterfactual_sents(self, bias_type):
        self._bias_type = bias_type
        self._augment_func = self._bias_type_to_func[self._bias_type]
        
        examples = []
            
        self.load_attributes(bias_type)

        for sentence in tqdm(
            self.tokenized_data,
            desc=f"Collecting counterfactual examples",
            leave=False
        ):
            sentence = sentence.lower()
            sentence = sentence.strip()
            examples = self._augment_func(self, sentence, examples, self._attribute_words)

        return examples