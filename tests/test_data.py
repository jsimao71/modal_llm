import torch

from modal_llm.data import ConstraintDataset, collate_examples


def test_generator_is_deterministic_and_near_miss_changes_one_facet():
    first = ConstraintDataset(4, seed=7, split="train")[2]
    second = ConstraintDataset(4, seed=7, split="train")[2]
    assert torch.equal(first["prompt"], second["prompt"])
    assert torch.equal(first["target"], second["target"])
    assert int(first["target"].ne(first["corrupted"]).sum()) == 1


def test_collate_pads_only_prompts():
    dataset = ConstraintDataset(8, seed=3, split="train")
    batch = collate_examples([dataset[0], dataset[1]])
    assert batch["prompt"].shape[0] == 2
    assert batch["target"].shape == (2, dataset.max_facets + 1)
