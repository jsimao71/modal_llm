"""Shared Transformer with explicit encode/generate/review/validate modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import Vocabulary


MODES = {"encode": 0, "generate": 1, "review": 2, "validate": 3}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    encode_mask: str | None
    encoded_prompt: bool
    use_goal: bool
    review: str | None
    review_prompt: bool
    validator: str | None
    matched_mlp: bool = False


BASELINES = {
    "B0": BaselineSpec("B0", None, False, False, None, False, None),
    "B1": BaselineSpec("B1", None, False, False, None, False, None, True),
    "B2": BaselineSpec("B2", "causal", True, False, None, False, None),
    "B3": BaselineSpec("B3", "bidirectional", True, False, None, False, None),
    "B4": BaselineSpec("B4", "causal", True, True, None, False, None),
    "B5": BaselineSpec("B5", "bidirectional", True, True, None, False, None),
    "B6": BaselineSpec("B6", "bidirectional", True, True, "final", False, "latent"),
    "B7": BaselineSpec("B7", "bidirectional", True, True, "causal", False, "latent"),
    "B8": BaselineSpec("B8", "bidirectional", True, True, "bidirectional", False, "latent"),
    "B9": BaselineSpec("B9", None, False, False, "bidirectional", True, "reread"),
    "B10": BaselineSpec("B10", "bidirectional", True, True, "bidirectional", True, "latent"),
}


class ModeTransformer(nn.Module):
    """All baselines instantiate the same parameters; only computation is ablated."""

    def __init__(
        self,
        vocab: Vocabulary,
        baseline: str,
        *,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        if baseline not in BASELINES:
            raise ValueError(f"Unknown baseline {baseline!r}; choose from {sorted(BASELINES)}")
        self.vocab = vocab
        self.spec = BASELINES[baseline]
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab.size, d_model, padding_idx=vocab.PAD)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.mode_embedding = nn.Embedding(len(MODES), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * ff_mult, dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.goal_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.review_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.goal_condition = nn.Linear(d_model, d_model, bias=False)
        self.match_control = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab.size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.facet_head = nn.Linear(d_model, vocab.max_facets)
        validator_width = d_model * 4
        self.validator_body = nn.Sequential(
            nn.Linear(validator_width, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.validator_score = nn.Linear(d_model, 1)
        self.validator_facets = nn.Linear(d_model, vocab.max_facets)
        self._forward_calls = 0
        self._token_positions = 0
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_compute_stats(self) -> None:
        self._forward_calls = 0
        self._token_positions = 0

    def compute_stats(self) -> dict[str, int]:
        return {"forward_calls": self._forward_calls, "token_positions": self._token_positions}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _embeddings(self, ids: torch.Tensor, mode: str, *, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        return (
            self.token_embedding(ids)
            + self.position_embedding(positions)[None, :, :]
            + self.mode_embedding.weight[MODES[mode]][None, None, :]
        )

    def _run(self, embeddings: torch.Tensor, padding: torch.Tensor, *, causal: bool) -> torch.Tensor:
        self._forward_calls += 1
        self._token_positions += int((~padding).sum())
        length = embeddings.shape[1]
        mask = None
        if causal:
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=embeddings.device), diagonal=1
            )
        return self.transformer(
            embeddings, mask=mask, src_key_padding_mask=padding, is_causal=causal
        )

    @staticmethod
    def _mean(hidden: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        active = (~padding).to(hidden.dtype).unsqueeze(-1)
        return (hidden * active).sum(1) / active.sum(1).clamp_min(1.0)

    def encode(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.encode_mask is None:
            raise RuntimeError(f"{self.spec.name} has no separate encode pass")
        padding = prompt.eq(self.vocab.PAD)
        hidden = self._run(
            self._embeddings(prompt, "encode"), padding,
            causal=self.spec.encode_mask == "causal",
        )
        goal = self.goal_head(self._mean(hidden, padding))
        return hidden, goal

    def generation_forward(
        self,
        prompt: torch.Tensor,
        decoder_input: torch.Tensor,
        *,
        forced_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        prompt_padding = prompt.eq(self.vocab.PAD)
        goal = None
        if self.spec.encode_mask is not None:
            encoded, goal = self.encode(prompt)
            if self.spec.encoded_prompt:
                prefix = encoded + self.mode_embedding.weight[MODES["generate"]][None, None, :]
            else:
                prefix = self._embeddings(prompt, "generate")
        else:
            prefix = self._embeddings(prompt, "generate")

        suffix = self._embeddings(decoder_input, "generate", offset=prompt.shape[1])
        if self.spec.matched_mlp:
            suffix = suffix + self.match_control(suffix)
        effective_goal = forced_goal if forced_goal is not None else goal
        if self.spec.use_goal and effective_goal is not None:
            conditioned = self.goal_condition(effective_goal).unsqueeze(1)
            prefix = prefix + conditioned
            suffix = suffix + conditioned
        embeddings = torch.cat([prefix, suffix], dim=1)
        padding = torch.cat([prompt_padding, decoder_input.eq(self.vocab.PAD)], dim=1)
        hidden = self._run(embeddings, padding, causal=True)
        generated_hidden = hidden[:, prompt.shape[1] :, :]
        return self.lm_head(generated_hidden), goal, generated_hidden

    def generate(
        self,
        prompt: torch.Tensor,
        max_tokens: int,
        *,
        forced_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        decoder = torch.full(
            (prompt.shape[0], 1), self.vocab.OUT_BOS, dtype=torch.long, device=prompt.device
        )
        goal = None
        calls = 0
        for _ in range(max_tokens):
            logits, goal, _ = self.generation_forward(prompt, decoder, forced_goal=forced_goal)
            decoder = torch.cat([decoder, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            calls += 1 + int(self.spec.encode_mask is not None)
        return decoder[:, 1:], goal, calls

    def _review_representations(
        self, prompt: torch.Tensor, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.spec.review == "final":
            decoder_input = torch.cat(
                [torch.full_like(candidate[:, :1], self.vocab.OUT_BOS), candidate[:, :-1]], dim=1
            )
            _, _, hidden = self.generation_forward(prompt, decoder_input)
            output = self.review_head(hidden[:, -1])
            return output, None

        if self.spec.review is None:
            raise RuntimeError(f"{self.spec.name} has no review pass")
        if self.spec.review_prompt:
            ids = torch.cat([prompt, candidate], dim=1)
            prompt_width = prompt.shape[1]
        else:
            ids = candidate
            prompt_width = 0
        padding = ids.eq(self.vocab.PAD)
        hidden = self._run(
            self._embeddings(ids, "review"), padding,
            causal=self.spec.review == "causal",
        )
        output_hidden = hidden[:, prompt_width:, :]
        output_padding = padding[:, prompt_width:]
        output = self.review_head(self._mean(output_hidden, output_padding))
        reread_goal = None
        if self.spec.validator == "reread":
            reread_goal = self.goal_head(self._mean(hidden[:, :prompt_width], padding[:, :prompt_width]))
        return output, reread_goal

    def validation_logits(
        self, prompt: torch.Tensor, candidate: torch.Tensor, goal: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, reread_goal = self._review_representations(prompt, candidate)
        if self.spec.validator == "reread":
            intended = reread_goal
        else:
            if goal is None:
                _, goal = self.encode(prompt)
            intended = goal
        assert intended is not None
        features = torch.cat(
            [intended, output, intended - output, intended * output], dim=-1
        )
        hidden = self.validator_body(features)
        return self.validator_score(hidden).squeeze(-1), self.validator_facets(hidden)

    def losses(self, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
        target = batch["target"]
        decoder_input = torch.cat(
            [torch.full_like(target[:, :1], self.vocab.OUT_BOS), target[:, :-1]], dim=1
        )
        logits, goal, _ = self.generation_forward(batch["prompt"], decoder_input)
        lm = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=self.vocab.PAD
        )
        total = weights.get("lm", 1.0) * lm
        result = {"lm": lm}

        if self.spec.use_goal and goal is not None:
            raw = F.binary_cross_entropy_with_logits(
                self.facet_head(goal), batch["bits"], reduction="none"
            )
            goal_loss = (raw * batch["active_facets"]).sum() / batch["active_facets"].sum()
            total = total + weights.get("goal", 0.2) * goal_loss
            result["goal"] = goal_loss

        if self.spec.validator is not None:
            positive, positive_facets = self.validation_logits(batch["prompt"], target, goal)
            negative, negative_facets = self.validation_logits(
                batch["prompt"], batch["corrupted"], goal
            )
            labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
            validation = F.binary_cross_entropy_with_logits(
                torch.cat([positive, negative]), labels
            )
            ranking = F.softplus(negative - positive).mean()
            active = batch["active_facets"]
            facet_positive = F.binary_cross_entropy_with_logits(
                positive_facets, active, reduction="none"
            )
            facet_negative = F.binary_cross_entropy_with_logits(
                negative_facets, batch["corrupt_satisfaction"], reduction="none"
            )
            facet = ((facet_positive + facet_negative) * active).sum() / (2 * active.sum())
            total = (
                total
                + weights.get("validation", 0.5) * validation
                + weights.get("ranking", 0.2) * ranking
                + weights.get("facet", 0.2) * facet
            )
            result.update(validation=validation, ranking=ranking, facet=facet)

        result["total"] = total
        return result
