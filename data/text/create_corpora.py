from pathlib import Path
import random
import re

class CorporaCreator:
    def __init__(
        self,
        delimiter: str = r"\n\s*\n",
        seed: int = 42,
    ):
        self.delimiter = delimiter
        self.seed = seed

    def count_articles(self, folder):
        count = 0

        for file in Path(folder).rglob("*"):
            if file.is_file():
                text = file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                docs = re.split(self.delimiter, text)
                count += sum(
                    1 for d in docs
                    if len(d.strip()) > 200
                )
        return count

    def reservoir_sample(self, folder, sample_size):
        random.seed(self.seed)
        reservoir = []
        seen = 0

        for file in Path(folder).rglob("*"):
            if not file.is_file():
                continue
            print("Processing:", file)
            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )
            docs = re.split(self.delimiter, text)

            for doc in docs:
                doc = doc.strip()
                if len(doc) <= 200:
                    continue
                seen += 1
                if len(reservoir) < sample_size:
                    reservoir.append(doc)
                else:
                    j = random.randint(0, seen-1)
                    if j < sample_size:
                        reservoir[j] = doc
        return reservoir

    def create_samples(self, language_folder, pct1 = 0.10, pct2 = 0.25):
        if pct1 > 1:
            pct1 = 1
        if pct2 > 1:
            pct2 = 1
        
        total = self.count_articles(language_folder)
        print("Total articles:", total)

        size_pct1 = int(total * pct1)

        sample_pct1 = self.reservoir_sample(
            language_folder,
            size_pct1
        )
        
        print(f"Sampling pct1={pct1*100}%:", size_pct1)

        random.seed(self.seed)
        random.shuffle(sample_pct1)

        size_pct2 = int(len(sample_pct1) * pct2)
        print(f"Sampling pct2={pct2*100}% of pct1:", size_pct2)
        sample_pct2 = sample_pct1[:size_pct2]

        return sample_pct1, sample_pct2