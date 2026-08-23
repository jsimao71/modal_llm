"""Controlled, machine-checkable multi-constraint task generators."""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


GENERATOR_VERSION = "independent-facets-v3"
PROMPT_FAMILIES = {"standard", "reordered", "interleaved", "paraphrase"}
CORRUPTION_FAMILIES = {"single_flip", "late_flip", "mixed"}
GOAL_PROMPT_STYLES = {"rendered", "canonical"}


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
    TEMPLATE_C: int = 11
    FILLER: int = 12
    CONTENT_OPEN: int = 13
    CONTENT_CLOSE: int = 14
    GOAL_OPEN: int = 15
    GOAL_CLOSE: int = 16

    @property
    def requirement_base(self) -> int:
        return 17

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
    """Deterministic tasks with split-specific prompt and corruption families."""

    DEFAULT_PROMPT_FAMILY = {
        "train": "standard",
        "validation": "reordered",
        "test": "interleaved",
    }
    DEFAULT_CORRUPTION_FAMILY = {
        "train": "single_flip",
        "validation": "late_flip",
        "test": "mixed",
    }

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
        prompt_family: str | None = None,
        corruption_family: str | None = None,
        goal_prompt_style: str = "rendered",
        direct_goal_exposure: float = 0.0,
        target_repetitions: int = 1,
        namespace: str | None = None,
    ) -> None:
        if split not in self.DEFAULT_PROMPT_FAMILY:
            raise ValueError(f"Unknown split {split!r}")
        self.size = size
        self.seed = seed
        self.max_facets = max_facets
        self.min_facets = min_facets
        self.max_distractors = max_distractors
        self.max_filler = max_filler
        self.split = split
        self.namespace = namespace or split
        if self.namespace not in self.DEFAULT_PROMPT_FAMILY:
            raise ValueError(f"Unknown RNG namespace {self.namespace!r}")
        self.prompt_family = prompt_family or self.DEFAULT_PROMPT_FAMILY[split]
        self.corruption_family = corruption_family or self.DEFAULT_CORRUPTION_FAMILY[split]
        self.goal_prompt_style = goal_prompt_style
        self.direct_goal_exposure = float(direct_goal_exposure)
        self.target_repetitions = int(target_repetitions)
        if self.prompt_family not in PROMPT_FAMILIES:
            raise ValueError(f"Unknown prompt family {self.prompt_family!r}")
        if self.corruption_family not in CORRUPTION_FAMILIES:
            raise ValueError(f"Unknown corruption family {self.corruption_family!r}")
        if self.goal_prompt_style not in GOAL_PROMPT_STYLES:
            raise ValueError(f"Unknown goal prompt style {self.goal_prompt_style!r}")
        if not 0.0 <= self.direct_goal_exposure <= 1.0:
            raise ValueError("direct_goal_exposure must be between 0 and 1")
        if self.target_repetitions < 1:
            raise ValueError("target_repetitions must be positive")
        if self.target_repetitions > 1 and self.min_facets != self.max_facets:
            raise ValueError("long-horizon tasks require min_facets == max_facets")
        self.vocab = Vocabulary(max_facets)

    def __len__(self) -> int:
        return self.size

    def _rng(self, index: int) -> random.Random:
        split_offset = {"train": 0, "validation": 10_000_019, "test": 20_000_033}[
            self.namespace
        ]
        return random.Random(self.seed * 1_000_003 + split_offset + index)

    def _direct_rng(self, index: int) -> random.Random:
        split_offset = {"train": 0, "validation": 10_000_019, "test": 20_000_033}[
            self.namespace
        ]
        return random.Random(self.seed * 1_000_003 + split_offset + index + 70_000_121)

    def _requirement_block(self, facet: int, value: int) -> list[int]:
        return [
            self.vocab.REQ_OPEN,
            self.vocab.requirement(facet, value),
            self.vocab.REQ_CLOSE,
        ]

    def _distractor_block(self, facet: int, value: int) -> list[int]:
        return [
            self.vocab.DISTRACTOR_OPEN,
            self.vocab.requirement(facet, 1 - value),
            self.vocab.DISTRACTOR_CLOSE,
        ]

    def _render_prompt(
        self,
        rng: random.Random,
        values: list[int],
        order: list[int],
        family: str,
    ) -> list[int]:
        if family == "paraphrase":
            family = "reordered"
            order = list(reversed(order))
        template = {
            "standard": self.vocab.TEMPLATE_A,
            "reordered": self.vocab.TEMPLATE_B,
            "interleaved": self.vocab.TEMPLATE_C,
        }[family]
        prompt = [self.vocab.TASK_BOS, template]
        late_facet = order[-1]
        early = order[:-1]
        distractor_facets = [rng.randrange(len(values)) for _ in range(rng.randint(0, self.max_distractors))]

        if family == "standard":
            for facet in early:
                prompt.extend(self._requirement_block(facet, values[facet]))
                prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler))
            for facet in distractor_facets:
                prompt.extend(self._distractor_block(facet, values[facet]))
                prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler))
        elif family == "reordered":
            prompt.extend([self.vocab.FILLER] * rng.randint(1, max(1, self.max_filler)))
            for facet in reversed(early):
                prompt.extend(self._requirement_block(facet, values[facet]))
            for facet in reversed(distractor_facets):
                prompt.extend([self.vocab.FILLER])
                prompt.extend(self._distractor_block(facet, values[facet]))
        else:  # held-out interleaving of locally conflicting blocks
            remaining = list(distractor_facets)
            for facet in early:
                if remaining:
                    distractor = remaining.pop(0)
                    prompt.extend(self._distractor_block(distractor, values[distractor]))
                prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler + 1))
                prompt.extend(self._requirement_block(facet, values[facet]))
            for facet in remaining:
                prompt.extend(self._distractor_block(facet, values[facet]))

        prompt.extend([self.vocab.FILLER] * rng.randint(0, self.max_filler))
        prompt.extend(self._requirement_block(late_facet, values[late_facet]))
        prompt.append(self.vocab.TASK_END)
        return prompt

    def _render_content(
        self,
        order: list[int],
        family: str,
        *,
        schedule: list[int] | None = None,
    ) -> list[int]:
        if family == "paraphrase":
            family = "reordered"
            order = list(reversed(order))
        template = {
            "standard": self.vocab.TEMPLATE_A,
            "reordered": self.vocab.TEMPLATE_B,
            "interleaved": self.vocab.TEMPLATE_C,
        }[family]
        content = [self.vocab.TASK_BOS, template, self.vocab.CONTENT_OPEN]
        facet_order = list(schedule) if schedule is not None else list(order)
        if schedule is None and family == "reordered":
            facet_order = list(reversed(facet_order))
        elif schedule is None and family == "interleaved":
            facet_order = list(facet_order[::2] + facet_order[1::2])
        for facet in facet_order:
            content.append(self.vocab.requirement(facet, 0))
        content.extend([self.vocab.CONTENT_CLOSE, self.vocab.TASK_END])
        return content

    def _render_generation_prompt(
        self,
        content: list[int],
        values: list[int],
        index: int,
    ) -> list[int]:
        if self.direct_goal_exposure == 0.0:
            return content
        rng = self._direct_rng(index)
        exposed = [
            facet
            for facet in range(len(values))
            if self.direct_goal_exposure == 1.0 or rng.random() < self.direct_goal_exposure
        ]
        if not exposed:
            return content
        direct_goal = [self.vocab.GOAL_OPEN]
        for facet in exposed:
            direct_goal.extend(self._requirement_block(facet, values[facet]))
        direct_goal.append(self.vocab.GOAL_CLOSE)
        return content[:2] + direct_goal + content[2:]

    def _render_full_scheduled_prompt(
        self,
        content: list[int],
        values: list[int],
    ) -> list[int]:
        direct_goal = [self.vocab.GOAL_OPEN]
        for facet in range(len(values)):
            direct_goal.extend(self._requirement_block(facet, values[facet]))
        direct_goal.append(self.vocab.GOAL_CLOSE)
        return content[:2] + direct_goal + content[2:]

    def _render_goal_only_prompt(
        self,
        rng: random.Random,
        values: list[int],
        order: list[int],
        family: str,
    ) -> list[int]:
        prompt = [self.vocab.TASK_BOS, self.vocab.GOAL_OPEN]
        prompt.extend(self._render_prompt(rng, values, order, family)[1:-1])
        prompt.extend([self.vocab.GOAL_CLOSE, self.vocab.TASK_END])
        return prompt

    def _render_canonical_goal_prompt(
        self,
        values: list[int],
        order: list[int],
    ) -> list[int]:
        prompt = [self.vocab.TASK_BOS, self.vocab.GOAL_OPEN]
        for facet in order:
            prompt.extend(self._requirement_block(facet, values[facet]))
        prompt.extend([self.vocab.GOAL_CLOSE, self.vocab.TASK_END])
        return prompt

    def _corrupt_candidate(
        self,
        rng: random.Random,
        target: list[int],
        values: list[int],
        late_facet: int,
        index: int,
    ) -> tuple[list[int], str]:
        family = self.corruption_family
        corruption_type = family
        if family == "mixed":
            corruption_type = ("single_flip", "double_flip", "truncate", "wrong_end")[index % 4]
        corrupted = list(target)
        if corruption_type == "late_flip":
            corrupted[late_facet] = self.vocab.answer(late_facet, 1 - values[late_facet])
        elif corruption_type == "single_flip":
            facet = rng.randrange(len(values))
            corrupted[facet] = self.vocab.answer(facet, 1 - values[facet])
        elif corruption_type == "double_flip":
            facets = rng.sample(range(len(values)), k=min(2, len(values)))
            for facet in facets:
                corrupted[facet] = self.vocab.answer(facet, 1 - values[facet])
        elif corruption_type == "truncate":
            start = rng.randrange(len(values))
            for facet in range(start, len(values)):
                corrupted[facet] = self.vocab.PAD
        elif corruption_type == "wrong_end":
            corrupted[-1] = self.vocab.OUT_BOS
        if corrupted == target:
            raise AssertionError("Corruption must change the candidate")
        return corrupted, corruption_type

    def _build(self, index: int, *, prompt_family: str | None = None) -> dict[str, Any]:
        rng = self._rng(index)
        k = rng.randint(self.min_facets, self.max_facets)
        values = [rng.randint(0, 1) for _ in range(k)]
        order = list(range(k))
        rng.shuffle(order)
        family = prompt_family or self.prompt_family
        prompt = self._render_prompt(rng, values, order, family)
        target_schedule: list[int] | None = None
        if self.target_repetitions > 1:
            target_schedule = []
            schedule_rng = self._direct_rng(index)
            for _ in range(self.target_repetitions):
                block = list(range(k))
                schedule_rng.shuffle(block)
                target_schedule.extend(block)
        content_prompt = self._render_content(order, family, schedule=target_schedule)
        generation_prompt = self._render_generation_prompt(content_prompt, values, index)
        if target_schedule is not None:
            prompt = self._render_full_scheduled_prompt(content_prompt, values)
        if self.goal_prompt_style == "canonical":
            canonical_order = list(range(k))
            goal_prompt = self._render_canonical_goal_prompt(values, canonical_order)
        else:
            goal_prompt = self._render_goal_only_prompt(rng, values, order, family)
        paraphrase_rng = random.Random(self.seed * 1_000_003 + index + 40_000_087)
        paraphrase_prompt = self._render_prompt(
            paraphrase_rng, values, order, "paraphrase"
        )
        if self.goal_prompt_style == "canonical":
            paraphrase_goal_prompt = self._render_canonical_goal_prompt(
                values, list(reversed(range(k)))
            )
        else:
            paraphrase_goal_prompt = self._render_goal_only_prompt(
                paraphrase_rng, values, order, "paraphrase"
            )

        if target_schedule is None:
            output_facets = list(range(k))
            target = [self.vocab.answer(facet, values[facet]) for facet in output_facets]
            target.extend([self.vocab.PAD] * (self.max_facets - k))
            target_facets = output_facets + [-1] * (self.max_facets - k)
            target_mask = [True] * k + [False] * (self.max_facets - k)
        else:
            output_facets = target_schedule
            target = [self.vocab.answer(facet, values[facet]) for facet in output_facets]
            target_facets = list(output_facets)
            target_mask = [True] * len(output_facets)
        target.append(self.vocab.OUT_END)
        target_facets.append(-1)
        target_mask.append(False)
        corrupted, corruption_type = self._corrupt_candidate(
            rng, target, values, order[-1], index
        )

        active = [1.0 if facet < k else 0.0 for facet in range(self.max_facets)]
        bits = [float(values[facet]) if facet < k else 0.0 for facet in range(self.max_facets)]
        corrupt_satisfaction = [
            float(corrupted[facet] == target[facet]) if facet < k else 0.0
            for facet in range(self.max_facets)
        ]

        counterfactual_facet = rng.randrange(k)
        counterfactual_prompt = list(prompt)
        counterfactual_goal_prompt = list(goal_prompt)
        original = self.vocab.requirement(counterfactual_facet, values[counterfactual_facet])
        replacement = self.vocab.requirement(counterfactual_facet, 1 - values[counterfactual_facet])
        for position in range(1, len(counterfactual_prompt) - 1):
            if (
                counterfactual_prompt[position - 1] == self.vocab.REQ_OPEN
                and counterfactual_prompt[position] == original
                and counterfactual_prompt[position + 1] == self.vocab.REQ_CLOSE
            ):
                counterfactual_prompt[position] = replacement
                break
        else:
            raise AssertionError("Authoritative counterfactual facet not found")
        for position in range(1, len(counterfactual_goal_prompt) - 1):
            if counterfactual_goal_prompt[position] == original:
                counterfactual_goal_prompt[position] = replacement
                break
        else:
            raise AssertionError("Counterfactual goal facet not found")
        counterfactual_target = list(target)
        for target_position, facet in enumerate(target_facets):
            if facet == counterfactual_facet:
                counterfactual_target[target_position] = self.vocab.answer(
                    counterfactual_facet, 1 - values[counterfactual_facet]
                )

        return {
            "id": f"{self.split}-{index}",
            "prompt": torch.tensor(prompt, dtype=torch.long),
            "generation_prompt": torch.tensor(generation_prompt, dtype=torch.long),
            "goal_prompt": torch.tensor(goal_prompt, dtype=torch.long),
            "paraphrase_prompt": torch.tensor(paraphrase_prompt, dtype=torch.long),
            "paraphrase_goal_prompt": torch.tensor(paraphrase_goal_prompt, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "corrupted": torch.tensor(corrupted, dtype=torch.long),
            "counterfactual_prompt": torch.tensor(counterfactual_prompt, dtype=torch.long),
            "counterfactual_goal_prompt": torch.tensor(counterfactual_goal_prompt, dtype=torch.long),
            "counterfactual_target": torch.tensor(counterfactual_target, dtype=torch.long),
            "target_facets": torch.tensor(target_facets, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "bits": torch.tensor(bits),
            "active_facets": torch.tensor(active),
            "corrupt_satisfaction": torch.tensor(corrupt_satisfaction),
            "corrupt_facet": counterfactual_facet,
            "corruption_type": corruption_type,
            "facet_count": k,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._build(index)

    def paraphrase(self, index: int) -> torch.Tensor:
        return self[index]["paraphrase_prompt"]

    def manifest(self) -> dict[str, Any]:
        return {
            "generator": GENERATOR_VERSION,
            "split": self.split,
            "namespace": self.namespace,
            "seed": self.seed,
            "size": self.size,
            "prompt_family": self.prompt_family,
            "corruption_family": self.corruption_family,
            "goal_prompt_style": self.goal_prompt_style,
            "direct_goal_exposure": self.direct_goal_exposure,
            "target_repetitions": self.target_repetitions,
            "max_facets": self.max_facets,
            "min_facets": self.min_facets,
            "max_distractors": self.max_distractors,
            "max_filler": self.max_filler,
        }

    def content_hash(self) -> str:
        """Hash materialized prompts, labels, and corruption identities in index order."""

        digest = hashlib.sha256()
        digest.update(repr(sorted(self.manifest().items())).encode("utf-8"))
        for index in range(len(self)):
            row = self[index]
            for key in (
                "prompt",
                "generation_prompt",
                "goal_prompt",
                "paraphrase_prompt",
                "paraphrase_goal_prompt",
                "target",
                "corrupted",
                "counterfactual_prompt",
                "counterfactual_goal_prompt",
                "counterfactual_target",
                "target_facets",
                "target_mask",
            ):
                tensor = row[key].contiguous()
                digest.update(len(tensor).to_bytes(4, "little"))
                digest.update(tensor.numpy().tobytes())
            digest.update(row["corruption_type"].encode("ascii"))
        return digest.hexdigest()


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(row["prompt"].numel() for row in rows)
    generation_width = max(row["generation_prompt"].numel() for row in rows)
    goal_width = max(row["goal_prompt"].numel() for row in rows)
    paraphrase_width = max(row["paraphrase_prompt"].numel() for row in rows)
    paraphrase_goal_width = max(row["paraphrase_goal_prompt"].numel() for row in rows)
    prompts = torch.zeros((len(rows), width), dtype=torch.long)
    generation_prompts = torch.zeros((len(rows), generation_width), dtype=torch.long)
    goal_prompts = torch.zeros((len(rows), goal_width), dtype=torch.long)
    paraphrase_prompts = torch.zeros((len(rows), paraphrase_width), dtype=torch.long)
    paraphrase_goal_prompts = torch.zeros((len(rows), paraphrase_goal_width), dtype=torch.long)
    counterfactual_prompts = torch.zeros((len(rows), width), dtype=torch.long)
    counterfactual_goal_prompts = torch.zeros((len(rows), goal_width), dtype=torch.long)
    for index, row in enumerate(rows):
        prompts[index, : row["prompt"].numel()] = row["prompt"]
        generation_prompts[index, : row["generation_prompt"].numel()] = row["generation_prompt"]
        goal_prompts[index, : row["goal_prompt"].numel()] = row["goal_prompt"]
        paraphrase_prompts[index, : row["paraphrase_prompt"].numel()] = row[
            "paraphrase_prompt"
        ]
        paraphrase_goal_prompts[
            index, : row["paraphrase_goal_prompt"].numel()
        ] = row["paraphrase_goal_prompt"]
        counterfactual_prompts[index, : row["counterfactual_prompt"].numel()] = row[
            "counterfactual_prompt"
        ]
        counterfactual_goal_prompts[
            index, : row["counterfactual_goal_prompt"].numel()
        ] = row["counterfactual_goal_prompt"]
    return {
        "ids": [row["id"] for row in rows],
        "corruption_types": [row["corruption_type"] for row in rows],
        "prompt": prompts,
        "generation_prompt": generation_prompts,
        "goal_prompt": goal_prompts,
        "paraphrase_prompt": paraphrase_prompts,
        "paraphrase_goal_prompt": paraphrase_goal_prompts,
        "target": torch.stack([row["target"] for row in rows]),
        "corrupted": torch.stack([row["corrupted"] for row in rows]),
        "counterfactual_prompt": counterfactual_prompts,
        "counterfactual_goal_prompt": counterfactual_goal_prompts,
        "counterfactual_target": torch.stack([row["counterfactual_target"] for row in rows]),
        "target_facets": torch.stack([row["target_facets"] for row in rows]),
        "target_mask": torch.stack([row["target_mask"] for row in rows]),
        "bits": torch.stack([row["bits"] for row in rows]),
        "active_facets": torch.stack([row["active_facets"] for row in rows]),
        "corrupt_satisfaction": torch.stack([row["corrupt_satisfaction"] for row in rows]),
        "corrupt_facet": torch.tensor([row["corrupt_facet"] for row in rows]),
        "facet_count": torch.tensor([row["facet_count"] for row in rows]),
    }
