import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
from pathlib import Path


class EmbeddingDataset(Dataset):
    """Loads JSONL triplets: {instruction, query, positive, negatives}"""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.samples = []
        path = Path(data_path)
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                self.samples = data[:max_samples] if max_samples else data
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        instruction = sample.get("instruction", "Represent the user's input.")
        query = sample["query"]
        positive = sample["positive"]
        negatives = sample.get("negatives", [])
        negative = random.choice(negatives) if negatives else None
        return {
            "instruction": instruction,
            "query": query,
            "positive": positive,
            "negative": negative,
        }


def collate_triplets(batch: List[Dict]) -> Dict[str, List[Dict]]:
    queries = []
    positives = []
    negatives = []
    for item in batch:
        q = {**item["query"], "instruction": item["instruction"]}
        p = {**item["positive"], "instruction": item["instruction"]}
        queries.append(q)
        positives.append(p)
        if item["negative"] is not None:
            n = {**item["negative"], "instruction": item["instruction"]}
            negatives.append(n)
    return {"queries": queries, "positives": positives, "negatives": negatives}


def build_dataloader(data_path: str, batch_size: int, max_samples: Optional[int] = None, num_workers: int = 0) -> DataLoader:
    dataset = EmbeddingDataset(data_path, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
        num_workers=num_workers,
        drop_last=True,
    )
