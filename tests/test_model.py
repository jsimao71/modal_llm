import torch

from modal_llm.data import ConstraintDataset, collate_examples
from modal_llm.model import ModeTransformer


def _batch():
    dataset = ConstraintDataset(4, seed=2, split="train")
    return dataset, collate_examples([dataset[0], dataset[1]])


def test_all_baselines_are_parameter_matched():
    dataset, _ = _batch()
    counts = {
        ModeTransformer(dataset.vocab, baseline, d_model=16, nhead=4, layers=1).parameter_count
        for baseline in ("B0", "B3", "B5", "B10")
    }
    assert len(counts) == 1


def test_causal_prefix_is_invariant_to_suffix_but_bidirectional_is_not():
    dataset, batch = _batch()
    model = ModeTransformer(
        dataset.vocab, "B5", d_model=16, nhead=4, layers=1, dropout=0.0
    ).eval()
    ids = batch["prompt"][:1].clone()
    alternative = ids.clone()
    nonpad = alternative[0].ne(dataset.vocab.PAD).nonzero().flatten()
    alternative[0, nonpad[-2]] = dataset.vocab.FILLER
    embeddings_a = model._embeddings(ids, "encode")
    embeddings_b = model._embeddings(alternative, "encode")
    padding = ids.eq(dataset.vocab.PAD)
    causal_a = model._run(embeddings_a, padding, causal=True)
    causal_b = model._run(embeddings_b, padding, causal=True)
    bidirectional_a = model._run(embeddings_a, padding, causal=False)
    bidirectional_b = model._run(embeddings_b, padding, causal=False)
    assert torch.allclose(causal_a[:, 0], causal_b[:, 0], atol=1e-6)
    assert not torch.allclose(bidirectional_a[:, 0], bidirectional_b[:, 0], atol=1e-6)


def test_b10_losses_are_finite_and_have_all_terms():
    dataset, batch = _batch()
    model = ModeTransformer(dataset.vocab, "B10", d_model=16, nhead=4, layers=1, dropout=0.0)
    losses = model.losses(batch, {})
    assert {"lm", "goal", "validation", "ranking", "facet", "total"} == set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
