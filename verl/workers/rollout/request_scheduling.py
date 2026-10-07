# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from verl.utils.import_utils import load_class_from_fqn


class RolloutRequestKind(str, Enum):
    """Semantic class of an agent rollout request."""

    FRESH = "fresh"
    CONTINUATION = "continuation"
    RETRY = "retry"


@dataclass(frozen=True)
class RolloutRequestContext:
    """Backend-neutral scheduling metadata for one rollout request attempt."""

    trajectory_id: str
    request_kind: RolloutRequestKind
    turn_index: int
    attempt_index: int
    base_priority: int
    prompt_tokens: int
    estimated_uncached_tokens: Optional[int]
    enqueued_at: float
    policy_version: Optional[int] = None
    expected_output_tokens: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        context = asdict(self)
        context["request_kind"] = self.request_kind.value
        return context

    @classmethod
    def from_dict(cls, context: dict[str, Any]) -> "RolloutRequestContext":
        """Build a typed context from its transport representation."""
        values = dict(context)
        values["request_kind"] = RolloutRequestKind(values["request_kind"])
        return cls(**values)


class RequestPriorityPolicy(Protocol):
    """Maps rollout request semantics to an optional backend priority."""

    def get_priority(self, context: RolloutRequestContext) -> Optional[int]:
        """Return a backend priority, or ``None`` to omit the priority hint."""
        ...


class RequestKindPriorityPolicy:
    """Configurable request-kind priority offsets for controlled experiments.

    This policy intentionally has no built-in offsets. It preserves the
    caller-provided base priority, which verl uses for stable ordering, and adds
    the configured request-kind offset.

    Args:
        priority_offsets: Mapping from ``fresh``, ``continuation``, or ``retry``
            to an integer offset.
    """

    def __init__(self, priority_offsets: dict[str, int]):
        valid_kinds = {kind.value for kind in RolloutRequestKind}
        unknown_kinds = set(priority_offsets) - valid_kinds
        if unknown_kinds:
            raise ValueError(f"Unknown rollout request kinds: {sorted(unknown_kinds)}")
        self.priority_offsets = {kind: int(offset) for kind, offset in priority_offsets.items()}

    def get_priority(self, context: RolloutRequestContext) -> Optional[int]:
        offset = self.priority_offsets.get(context.request_kind.value)
        if offset is None:
            return None
        return context.base_priority + offset


class LinearRequestCostEstimator:
    """Estimate request work from tokens already known by the agent loop.

    The estimate is intentionally lightweight and only needs to preserve a
    useful ordering between requests. Runtime measurements can tune the
    coefficients without changing scheduling semantics.

    Args:
        prefill_token_weight: Cost assigned to each estimated uncached token.
        decode_token_weight: Base cost assigned to each expected output token.
        decode_context_scale: Context length that adds one unit of decode cost.
        minimum_cost: Positive floor used by aging calculations.
    """

    def __init__(
        self,
        prefill_token_weight: float = 1.0,
        decode_token_weight: float = 1.0,
        decode_context_scale: float = 8192.0,
        minimum_cost: float = 1.0,
    ):
        for name, value in {
            "prefill_token_weight": prefill_token_weight,
            "decode_token_weight": decode_token_weight,
            "decode_context_scale": decode_context_scale,
            "minimum_cost": minimum_cost,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.prefill_token_weight = float(prefill_token_weight)
        self.decode_token_weight = float(decode_token_weight)
        self.decode_context_scale = float(decode_context_scale)
        self.minimum_cost = float(minimum_cost)

    def estimate(self, context: RolloutRequestContext) -> float:
        uncached_tokens = context.estimated_uncached_tokens
        if uncached_tokens is None:
            uncached_tokens = context.prompt_tokens
        uncached_tokens = max(0, uncached_tokens)
        output_tokens = max(0, context.expected_output_tokens or 0)
        context_multiplier = 1.0 + max(0, context.prompt_tokens) / self.decode_context_scale
        estimated_cost = (
            self.prefill_token_weight * uncached_tokens + self.decode_token_weight * output_tokens * context_multiplier
        )
        return max(self.minimum_cost, estimated_cost)


def effective_request_cost(
    context: RolloutRequestContext,
    *,
    wait_seconds: float,
    target_wait_seconds: float,
    cost_estimator: LinearRequestCostEstimator,
) -> float:
    """Apply soft aging to an estimated request cost."""
    if target_wait_seconds <= 0:
        raise ValueError("target_wait_seconds must be positive")
    return cost_estimator.estimate(context) / (1.0 + max(0.0, wait_seconds) / target_wait_seconds)


class EffectiveCostPriorityPolicy:
    """Map the effective-cost ordering to bounded vLLM priorities.

    Lower effective cost maps to a lower integer priority. Requests that cross
    ``max_wait_seconds`` use ``overdue_priority`` as a hard starvation guard.

    Args:
        target_wait_seconds: Aging time scale in the effective-cost formula.
        cost_per_priority: Effective-cost units represented by one priority.
        min_priority: Lowest generated non-overdue priority.
        max_priority: Highest generated priority.
        max_wait_seconds: Optional hard waiting-time limit.
        overdue_priority: Priority used after the hard wait limit.
        priority_stride: Scale separating cost buckets so ``base_priority`` is
            only a tie-breaker and cannot reverse cost ordering.
        cost_estimator_kwargs: Arguments for :class:`LinearRequestCostEstimator`.
    """

    def __init__(
        self,
        target_wait_seconds: float,
        cost_per_priority: float,
        min_priority: int = 0,
        max_priority: int = 16,
        max_wait_seconds: Optional[float] = None,
        overdue_priority: int = -1,
        priority_stride: int = 1_000_000,
        cost_estimator_kwargs: Optional[dict[str, Any]] = None,
    ):
        if target_wait_seconds <= 0:
            raise ValueError("target_wait_seconds must be positive")
        if cost_per_priority <= 0:
            raise ValueError("cost_per_priority must be positive")
        if min_priority > max_priority:
            raise ValueError("min_priority must not exceed max_priority")
        if max_wait_seconds is not None and max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive or null")
        if not isinstance(priority_stride, int) or isinstance(priority_stride, bool) or priority_stride <= 0:
            raise ValueError("priority_stride must be a positive integer")
        self.target_wait_seconds = float(target_wait_seconds)
        self.cost_per_priority = float(cost_per_priority)
        self.min_priority = int(min_priority)
        self.max_priority = int(max_priority)
        self.max_wait_seconds = max_wait_seconds
        self.overdue_priority = int(overdue_priority)
        self.priority_stride = priority_stride
        self.cost_estimator = LinearRequestCostEstimator(**(cost_estimator_kwargs or {}))
        self._clock = time.time

    def get_priority(self, context: RolloutRequestContext) -> Optional[int]:
        wait_seconds = max(0.0, self._clock() - context.enqueued_at)
        if self.max_wait_seconds is not None and wait_seconds >= self.max_wait_seconds:
            return self.overdue_priority * self.priority_stride + context.base_priority
        effective_cost = effective_request_cost(
            context,
            wait_seconds=wait_seconds,
            target_wait_seconds=self.target_wait_seconds,
            cost_estimator=self.cost_estimator,
        )
        priority = self.min_priority + math.floor(effective_cost / self.cost_per_priority)
        priority = min(self.max_priority, max(self.min_priority, priority))
        return priority * self.priority_stride + context.base_priority


def load_request_priority_policy(
    policy_class: Optional[str],
    policy_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[RequestPriorityPolicy]:
    """Instantiate a configured request-priority policy."""
    if not policy_class:
        return None

    cls = load_class_from_fqn(policy_class, "request priority policy")
    policy = cls(**(policy_kwargs or {}))
    if not callable(getattr(policy, "get_priority", None)):
        raise TypeError(f"Request priority policy '{policy_class}' must define get_priority(context).")
    return policy
