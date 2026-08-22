"""Controlled, machine-checkable multi-constraint task generators."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


GENERATOR_VERSION = "independent-facets-v1"


@dataclass(frozen=True)
class Vocabulary:
    max_facets: int

    PAD: int = 0
    TASK_BOS: int = 1
    TASK_END: int = 2
    OUT_BOS: int = 3
    OUT_END: int = 4
    REQ_OPEN: int = 5
    REQ_CLOSE: int = 6
    DISTRACTOR_OPEN: int = 7
    DISTRACTOR_CLOSE: int = 8
    TEMPLATE_A: int = 9
    TEMPLATE_B: int = 10
    FILLER: int = 11

    @property
    def requirement_base(self) -> int:
        return 12

    @property
    def answer_base(self) -> int:
        return self.requirement_base + 2 * self.max_facets

    @property
    def size(self) -> int:
        return self.answer_base + 2 * self.max_facets

    def requirement(self, facet: int, value: int) -> int:
        return self.requirement_base + 2 * facet + value

    def answer(self, facet: int, value: int) -> int:
        return self.answer_base + 2 * facet + value


class ConstraintDataset(Dataset):
    """Deterministic tasks with late constraints, distractors, and near misses."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        max_facets: int = 4,
        min_facets: int = 2,
        max_distractors: int = 2,
        max_filler: int = 3,
        split: str = "train",
    ) -> None:
        self.size = size
        self.seed = seed
        self.max_facets = max_facets
        self.min_facets = min_facets
        self.max_distractors = max_distractors
        self.max_filler = max_filler
        self.split = split
        self.vocab = Vocabulary(max_facets)

    def __len__(self) -> int:
        return self.size

    def _build(self, index: int, *, paraphrase: bool = False) -> dict[str, Any]:
        split_offset = {"train": 0, "validation": 10_000_019, "test": 20_000_033}[self.split]
        rng = random.Random(self.seed * 1_000_003 + split_offset + index)
        k = rng.randint(self.min_facets, self.max_facets)
        values = [rng.randint(0, 1) for _ in range(k)]
        order = list(range(k))
        rng.shuffle(order)
        if paraphrase:
            order.reverse()

        template = self.vocab.TEMPLATE_B if paraphrase else self.vocab.TEMPLATE_A
        prompt = [self.vocab.TASK_BOS, template]
        late_facet = order[-1]
        for facet in order:
            if facet == late_facet:
                continue
            prompt.extend(
                [self.vocab.REQ_OPEN, self.vocab.requirement(facet, values[facet]), self.vocab.REQ_CLOSE]
            )
            prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler))

        distractors = rng.randint(0, self.max_distractors)
        for _ in range(distractors):
            facet = rng.randrange(k)
            prompt.extend(
                [self.vocab.DISTRACTOR_OPEN,
                 self.vocab.requirement(facet, 1 - values[facet]),
                 self.vocab.DISTRACTOR_CLOSE]
            )
            prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler))

        prompt.extend(
            [self.vocab.REQ_OPEN,
             self.vocab.requirement(late_facet, values[late_facet]),
             self.vocab.REQ_CLOSE,
             self.vocab.TASK_END]
        )
        target = [self.vocab.answer(i, values[i]) for i in range(k)]
        target.extend([self.vocab.PAD] * (self.max_facets - k))
        target.append(self.vocab.OUT_END)

        corrupt_facet = rng.randrange(k)
        corrupted = list(target)
        corrupted[corrupt_facet] = self.vocab.answer(corrupt_facet, 1 - values[corrupt_facet])
        active = [1.0 if i < k else 0.0 for i in range(self.max_facets)]
        bits = [float(values[i]) if i < k else 0.0 for i in range(self.max_facets)]
        corrupt_satisfaction = [1.0 if i < k else 0.0 for i in range(self.max_facets)]
        corrupt_satisfaction[corrupt_facet] = 0.0
        counterfactual_prompt = list(prompt)
        original_requirement = self.vocab.requirement(corrupt_facet, values[corrupt_facet])
        replacement_requirement = self.vocab.requirement(corrupt_facet, 1 - values[corrupt_facet])
        for position in range(1, len(counterfactual_prompt) - 1):
            if (
                counterfactual_prompt[position - 1] == self.vocab.REQ_OPEN
                and counterfactual_prompt[position] == original_requirement
                and counterfactual_prompt[position + 1] == self.vocab.REQ_CLOSE
            ):
                counterfactual_prompt[position] = replacement_requirement
                break
        counterfactual_target = list(target)
        counterfactual_target[corrupt_facet] = self.vocab.answer(
            corrupt_facet, 1 - values[corrupt_facet]
        )
        return {
            "id": f"{self.split}-{index}",
            "prompt": torch.tensor(prompt, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "corrupted": torch.tensor(corrupted, dtype=torch.long),
            "counterfactual_prompt": torch.tensor(counterfactual_prompt, dtype=torch.long),
            "counterfactual_target": torch.tensor(counterfactual_target, dtype=torch.long),
            "bits": torch.tensor(bits),
            "active_facets": torch.tensor(active),
            "corrupt_satisfaction": torch.tensor(corrupt_satisfaction),
            "corrupt_facet": corrupt_facet,
            "facet_count": k,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._build(index)

    def paraphrase(self, index: int) -> torch.Tensor:
        return self._build(index, paraphrase=True)["prompt"]

    def manifest(self) -> dict[str, Any]:
        return {
            "generator": GENERATOR_VERSION,
            "split": self.split,
            "seed": self.seed,
            "size": self.size,
            "max_facets": self.max_facets,
            "min_facets": self.min_facets,
            "max_distractors": self.max_distractors,
            "max_filler": self.max_filler,
        }


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(row["prompt"].numel() for row in rows)
    prompts = torch.zeros((len(rows), width), dtype=torch.long)
    counterfactual_prompts = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        prompts[index, : row["prompt"].numel()] = row["prompt"]
        counterfactual_prompts[index, : row["counterfactual_prompt"].numel()] = row[
            "counterfactual_prompt"
        ]
    return {
        "ids": [row["id"] for row in rows],
        "prompt": prompts,
        "target": torch.stack([row["target"] for row in rows]),
        "corrupted": torch.stack([row["corrupted"] for row in rows]),
        "counterfactual_prompt": counterfactual_prompts,
        "counterfactual_target": torch.stack([row["counterfactual_target"] for row in rows]),
        "bits": torch.stack([row["bits"] for row in rows]),
        "active_facets": torch.stack([row["active_facets"] for row in rows]),
        "corrupt_satisfaction": torch.stack([row["corrupt_satisfaction"] for row in rows]),
        "corrupt_facet": torch.tensor([row["corrupt_facet"] for row in rows]),
        "facet_count": torch.tensor([row["facet_count"] for row in rows]),
    }
