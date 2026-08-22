import torch

from modal_llm.common import seed_everything
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


def test_active_parameter_counts_reflect_enabled_components():
    dataset, _ = _batch()
    models = {
        baseline: ModeTransformer(dataset.vocab, baseline, d_model=16, nhead=4, layers=1)
        for baseline in ("B0", "B1", "B5", "B10")
    }
    assert models["B0"].active_parameter_count < models["B1"].active_parameter_count
    assert models["B1"].active_parameter_count < models["B5"].active_parameter_count
    assert models["B5"].active_parameter_count < models["B10"].active_parameter_count


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


def test_cached_generation_context_matches_fresh_forward_and_avoids_reencoding():
    dataset, batch = _batch()
    model = ModeTransformer(
        dataset.vocab, "B5", d_model=16, nhead=4, layers=1, dropout=0.0
    ).eval()
    decoder = torch.full((2, 1), dataset.vocab.OUT_BOS, dtype=torch.long)
    model.reset_compute_stats()
    fresh, _, _ = model.generation_forward(batch["prompt"], decoder)
    fresh_calls = model.compute_stats()["forward_calls"]
    model.reset_compute_stats()
    context = model.prepare_generation(batch["prompt"])
    cached, _, _ = model.generation_forward(batch["prompt"], decoder, context=context)
    cached_calls = model.compute_stats()["forward_calls"]
    assert torch.allclose(fresh, cached, atol=1e-6)
    assert fresh_calls == cached_calls == 2

    model.reset_compute_stats()
    model.generate(batch["prompt"], max_tokens=3)
    assert model.compute_stats()["forward_calls"] == 4  # one encode plus three decode passes


def test_seeded_initialization_and_optimizer_step_are_deterministic():
    dataset, batch = _batch()

    def one_step():
        seed_everything(29)
        model = ModeTransformer(
            dataset.vocab, "B5", d_model=16, nhead=4, layers=1, dropout=0.0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        model.losses(batch, {})["total"].backward()
        optimizer.step()
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    first = one_step()
    second = one_step()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
