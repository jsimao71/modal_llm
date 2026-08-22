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


@dataclass(frozen=True)
class GenerationContext:
    """Prompt-side state computed once and latched for autoregressive decoding."""

    prefix: torch.Tensor
    prompt_padding: torch.Tensor
    prefix_length: int
    goal: torch.Tensor | None
    conditioning: torch.Tensor | None


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
        generation_prompt_only: bool = False,
        goal_vectors: int = 1,
    ) -> None:
        super().__init__()
        if baseline not in BASELINES:
            raise ValueError(f"Unknown baseline {baseline!r}; choose from {sorted(BASELINES)}")
        self.vocab = vocab
        self.spec = BASELINES[baseline]
        self.d_model = d_model
        self.generation_prompt_only = generation_prompt_only
        self.goal_vectors = goal_vectors
        self.latent_width = d_model * goal_vectors
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
        self.goal_queries = nn.Parameter(torch.randn(goal_vectors, d_model) * 0.02)
        self.review_queries = nn.Parameter(torch.randn(goal_vectors, d_model) * 0.02)
        self.goal_condition = nn.Linear(self.latent_width, d_model, bias=False)
        self.match_control = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab.size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.facet_head = nn.Linear(self.latent_width, vocab.max_facets)
        validator_width = self.latent_width * 4
        self.validator_body = nn.Sequential(
            nn.Linear(validator_width, self.latent_width),
            nn.GELU(),
            nn.LayerNorm(self.latent_width),
        )
        self.validator_score = nn.Linear(self.latent_width, 1)
        self.validator_facets = nn.Linear(self.latent_width, vocab.max_facets)
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

    @property
    def active_parameter_count(self) -> int:
        modules: list[nn.Module] = [
            self.token_embedding,
            self.position_embedding,
            self.mode_embedding,
            self.transformer,
            self.lm_head,
        ]
        if self.spec.matched_mlp:
            modules.append(self.match_control)
        if self.spec.use_goal:
            modules.extend([self.goal_head, self.goal_condition, self.facet_head])
        if self.spec.review is not None:
            modules.append(self.review_head)
        if self.spec.validator is not None:
            modules.extend([self.validator_body, self.validator_score, self.validator_facets])
        if self.spec.validator == "reread":
            modules.append(self.goal_head)
        unique = {id(parameter): parameter for module in modules for parameter in module.parameters()}
        return sum(parameter.numel() for parameter in unique.values())

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

    @staticmethod
    def _flatten_state(state: torch.Tensor) -> torch.Tensor:
        return state.reshape(state.shape[0], -1)

    def _extract_state(
        self,
        hidden: torch.Tensor,
        padding: torch.Tensor,
        *,
        queries: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        if self.goal_vectors == 1:
            return head(self._mean(hidden, padding))
        mask = padding.unsqueeze(1)
        scores = torch.einsum("kd,btd->bkt", queries, hidden) / (self.d_model ** 0.5)
        scores = scores.masked_fill(mask, -1e9)
        weights = scores.softmax(-1)
        contexts = torch.einsum("bkt,btd->bkd", weights, hidden)
        return head(contexts)

    def encode(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.encode_mask is None:
            raise RuntimeError(f"{self.spec.name} has no separate encode pass")
        padding = prompt.eq(self.vocab.PAD)
        hidden = self._run(
            self._embeddings(prompt, "encode"), padding,
            causal=self.spec.encode_mask == "causal",
        )
        goal = self._extract_state(
            hidden, padding, queries=self.goal_queries, head=self.goal_head
        )
        return hidden, goal

    def prepare_generation(
        self,
        prompt: torch.Tensor,
        *,
        goal_prompt: torch.Tensor | None = None,
        forced_goal: torch.Tensor | None = None,
    ) -> GenerationContext:
        """Compute prompt features and latch goal conditioning exactly once."""

        prompt_padding = prompt.eq(self.vocab.PAD)
        goal = None
        goal_prompt = prompt if goal_prompt is None else goal_prompt
        if self.spec.encode_mask is not None:
            encoded, goal = self.encode(goal_prompt)
            if self.spec.encoded_prompt and not self.generation_prompt_only:
                prefix = encoded + self.mode_embedding.weight[MODES["generate"]][None, None, :]
                prefix_padding = goal_prompt.eq(self.vocab.PAD)
            else:
                prefix = self._embeddings(prompt, "generate")
                prefix_padding = prompt_padding
        else:
            prefix = self._embeddings(prompt, "generate")
            prefix_padding = prompt_padding
        effective_goal = forced_goal if forced_goal is not None else goal
        conditioning = None
        if self.spec.use_goal and effective_goal is not None:
            conditioning = self.goal_condition(self._flatten_state(effective_goal)).unsqueeze(1)
            prefix = prefix + conditioning
        return GenerationContext(prefix, prefix_padding, prefix.shape[1], goal, conditioning)

    def generation_forward(
        self,
        prompt: torch.Tensor,
        decoder_input: torch.Tensor,
        *,
        goal_prompt: torch.Tensor | None = None,
        forced_goal: torch.Tensor | None = None,
        context: GenerationContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if context is not None and forced_goal is not None:
            raise ValueError("Pass forced_goal when preparing context, not with an existing context")
        if context is not None and goal_prompt is not None:
            raise ValueError("Pass goal_prompt when preparing context, not with an existing context")
        context = context or self.prepare_generation(
            prompt, goal_prompt=goal_prompt, forced_goal=forced_goal
        )
        suffix = self._embeddings(decoder_input, "generate", offset=context.prefix_length)
        if self.spec.matched_mlp:
            suffix = suffix + self.match_control(suffix)
        if context.conditioning is not None:
            suffix = suffix + context.conditioning
        embeddings = torch.cat([context.prefix, suffix], dim=1)
        padding = torch.cat([context.prompt_padding, decoder_input.eq(self.vocab.PAD)], dim=1)
        hidden = self._run(embeddings, padding, causal=True)
        generated_hidden = hidden[:, context.prefix_length :, :]
        return self.lm_head(generated_hidden), context.goal, generated_hidden

    def generate(
        self,
        prompt: torch.Tensor,
        max_tokens: int,
        *,
        goal_prompt: torch.Tensor | None = None,
        forced_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        decoder = torch.full(
            (prompt.shape[0], 1), self.vocab.OUT_BOS, dtype=torch.long, device=prompt.device
        )
        before = self.compute_stats()["forward_calls"]
        context = self.prepare_generation(prompt, goal_prompt=goal_prompt, forced_goal=forced_goal)
        for _ in range(max_tokens):
            logits, _, _ = self.generation_forward(prompt, decoder, context=context)
            decoder = torch.cat([decoder, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        calls = self.compute_stats()["forward_calls"] - before
        return decoder[:, 1:], context.goal, calls

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
        output = self._extract_state(
            output_hidden, output_padding, queries=self.review_queries, head=self.review_head
        )
        reread_goal = None
        if self.spec.validator == "reread":
            reread_goal = self._extract_state(
                hidden[:, :prompt_width],
                padding[:, :prompt_width],
                queries=self.goal_queries,
                head=self.goal_head,
            )
        return output, reread_goal

    def validation_logits(
        self,
        prompt: torch.Tensor,
        candidate: torch.Tensor,
        goal: torch.Tensor | None = None,
        *,
        goal_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, reread_goal = self._review_representations(prompt, candidate)
        if self.spec.validator == "reread":
            intended = reread_goal
        else:
            if goal is None:
                _, goal = self.encode(prompt if goal_prompt is None else goal_prompt)
            intended = goal
        assert intended is not None
        intended_flat = self._flatten_state(intended)
        output_flat = self._flatten_state(output)
        features = torch.cat(
            [
                intended_flat,
                output_flat,
                intended_flat - output_flat,
                intended_flat * output_flat,
            ],
            dim=-1,
        )
        hidden = self.validator_body(features)
        return self.validator_score(hidden).squeeze(-1), self.validator_facets(hidden)

    def _goal_losses(
        self,
        goal: torch.Tensor,
        batch: dict[str, torch.Tensor],
        invariance_weight: float,
    ) -> dict[str, torch.Tensor]:
        raw = F.binary_cross_entropy_with_logits(
            self.facet_head(self._flatten_state(goal)), batch["bits"], reduction="none"
        )
        facet = (raw * batch["active_facets"]).sum() / batch["active_facets"].sum()
        result = {"goal": facet}
        if invariance_weight > 0:
            goal_prompt = batch.get("paraphrase_goal_prompt", batch["paraphrase_prompt"])
            _, paraphrase_goal = self.encode(goal_prompt)
            invariance = (
                1.0 - F.cosine_similarity(goal, paraphrase_goal, dim=-1)
            ).mean()
            result["goal_invariance"] = invariance
        return result

    def goal_objective(
        self,
        batch: dict[str, torch.Tensor],
        *,
        invariance_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if not self.spec.use_goal:
            raise RuntimeError(f"{self.spec.name} does not use a goal state")
        _, goal = self.encode(batch.get("goal_prompt", batch["prompt"]))
        losses = self._goal_losses(goal, batch, invariance_weight)
        total = losses["goal"] + invariance_weight * losses.get(
            "goal_invariance", goal.new_zeros(())
        )
        return {**losses, "total": total}

    def losses(self, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
        target = batch["target"]
        generation_prompt = batch.get("generation_prompt", batch["prompt"])
        goal_prompt = batch.get("goal_prompt", batch["prompt"])
        validation_prompt = batch["prompt"]
        decoder_input = torch.cat(
            [torch.full_like(target[:, :1], self.vocab.OUT_BOS), target[:, :-1]], dim=1
        )
        logits, goal, _ = self.generation_forward(
            generation_prompt, decoder_input, goal_prompt=goal_prompt
        )
        lm = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=self.vocab.PAD
        )
        total = weights.get("lm", 1.0) * lm
        result = {"lm": lm}

        if self.spec.use_goal and goal is not None:
            goal_losses = self._goal_losses(
                goal, batch, float(weights.get("goal_invariance", 0.0))
            )
            total = total + weights.get("goal", 0.2) * goal_losses["goal"]
            if "goal_invariance" in goal_losses:
                total = total + weights.get("goal_invariance", 0.0) * goal_losses[
                    "goal_invariance"
                ]
            result.update(goal_losses)

        if self.spec.validator is not None:
            positive, positive_facets = self.validation_logits(
                validation_prompt, target, goal, goal_prompt=goal_prompt
            )
            negative, negative_facets = self.validation_logits(
                validation_prompt, batch["corrupted"], goal, goal_prompt=goal_prompt
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
