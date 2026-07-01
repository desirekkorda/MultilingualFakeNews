
import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len=384
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].flatten(),

            "attention_mask":
                encoding["attention_mask"].flatten(),

            "label":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }
