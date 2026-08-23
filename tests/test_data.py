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
    assert batch["generation_prompt"].shape[0] == 2
    assert batch["goal_prompt"].shape[0] == 2


def test_examples_expose_separate_generation_and_goal_prompts():
    row = ConstraintDataset(4, seed=5, split="train")[0]
    assert row["generation_prompt"].numel() < row["prompt"].numel()
    assert row["goal_prompt"].numel() <= row["prompt"].numel()
    assert not torch.equal(row["generation_prompt"], row["goal_prompt"])
    assert int(row["goal_prompt"].ne(row["counterfactual_goal_prompt"]).sum()) == 1


def test_canonical_goal_prompt_contains_only_authoritative_requirements():
    dataset = ConstraintDataset(
        4, seed=5, split="test", goal_prompt_style="canonical"
    )
    row = dataset[0]
    goal = row["goal_prompt"].tolist()
    paraphrase = row["paraphrase_goal_prompt"].tolist()
    forbidden = {
        dataset.vocab.DISTRACTOR_OPEN,
        dataset.vocab.DISTRACTOR_CLOSE,
        dataset.vocab.FILLER,
        dataset.vocab.TEMPLATE_A,
        dataset.vocab.TEMPLATE_B,
        dataset.vocab.TEMPLATE_C,
    }

    assert forbidden.isdisjoint(goal)
    assert goal.count(dataset.vocab.REQ_OPEN) == row["facet_count"]
    assert goal.count(dataset.vocab.REQ_CLOSE) == row["facet_count"]
    assert goal != paraphrase
    assert int(row["goal_prompt"].ne(row["counterfactual_goal_prompt"]).sum()) == 1


def test_splits_use_held_out_prompt_and_corruption_families():
    train = ConstraintDataset(8, seed=3, split="train")
    validation = ConstraintDataset(8, seed=3, split="validation")
    test = ConstraintDataset(8, seed=3, split="test")
    assert {train.prompt_family, validation.prompt_family, test.prompt_family} == {
        "standard", "reordered", "interleaved"
    }
    assert {train.corruption_family, validation.corruption_family, test.corruption_family} == {
        "single_flip", "late_flip", "mixed"
    }
    assert {test[index]["corruption_type"] for index in range(8)} == {
        "single_flip", "double_flip", "truncate", "wrong_end"
    }


def test_counterfactual_changes_one_authoritative_requirement_and_target():
    row = ConstraintDataset(4, seed=11, split="test")[0]
    assert int(row["prompt"].ne(row["counterfactual_prompt"]).sum()) == 1
    assert int(row["target"].ne(row["counterfactual_target"]).sum()) == 1


def test_dataset_content_hash_is_stable_and_split_sensitive():
    first = ConstraintDataset(4, seed=13, split="train")
    same = ConstraintDataset(4, seed=13, split="train")
    held_out = ConstraintDataset(4, seed=13, split="test")
    assert first.content_hash() == same.content_hash()
    assert first.content_hash() != held_out.content_hash()


def test_dataset_content_hash_covers_goal_prompt_style():
    rendered = ConstraintDataset(4, seed=13, split="train")
    canonical = ConstraintDataset(
        4, seed=13, split="train", goal_prompt_style="canonical"
    )

    assert rendered.content_hash() != canonical.content_hash()
