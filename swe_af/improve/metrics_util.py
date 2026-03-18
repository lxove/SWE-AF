"""Utilities for extracting and aggregating token usage metrics."""

from __future__ import annotations

from typing import Any

from swe_af.improve.schemas import ModelUsage, UsageSummary


def extract_step_metrics(response: Any, model: str) -> dict:
    """Extract a lightweight metrics dict from an AgentResponse.

    Returns a dict with keys: model, input_tokens, output_tokens, cost_usd.
    Safe to call even if metrics are unavailable.
    """
    try:
        m = response.metrics
        usage = m.usage or {}
        return {
            "model": model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": m.total_cost_usd or 0.0,
        }
    except Exception:
        return {"model": model, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


class UsageAccumulator:
    """Accumulates token usage across multiple AI calls, grouped by model."""

    def __init__(self) -> None:
        # model -> {input_tokens, output_tokens, cost_usd, num_calls}
        self._by_model: dict[str, dict[str, float]] = {}

    def add(self, step_metrics: dict | None) -> None:
        """Add metrics from a single step (the _metrics dict from sub-agents)."""
        if not step_metrics:
            return
        model = step_metrics.get("model", "unknown")
        if model not in self._by_model:
            self._by_model[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "num_calls": 0,
            }
        entry = self._by_model[model]
        entry["input_tokens"] += step_metrics.get("input_tokens", 0)
        entry["output_tokens"] += step_metrics.get("output_tokens", 0)
        entry["cost_usd"] += step_metrics.get("cost_usd", 0.0)
        entry["num_calls"] += 1

    def to_summary(self) -> UsageSummary:
        """Build a UsageSummary from accumulated data."""
        by_model = []
        total_in = 0
        total_out = 0
        total_cost = 0.0
        for model, data in sorted(self._by_model.items()):
            by_model.append(
                ModelUsage(
                    model=model,
                    input_tokens=int(data["input_tokens"]),
                    output_tokens=int(data["output_tokens"]),
                    cost_usd=round(data["cost_usd"], 6),
                    num_calls=int(data["num_calls"]),
                )
            )
            total_in += int(data["input_tokens"])
            total_out += int(data["output_tokens"])
            total_cost += data["cost_usd"]
        return UsageSummary(
            by_model=by_model,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost_usd=round(total_cost, 6),
        )

    def format_summary_line(self) -> str:
        """Return a single-line human-readable summary of usage."""
        s = self.to_summary()
        if not s.by_model:
            return ""
        parts = []
        for m in s.by_model:
            parts.append(
                f"{m.model}: {m.input_tokens:,}in/{m.output_tokens:,}out "
                f"(${m.cost_usd:.4f}, {m.num_calls} calls)"
            )
        total = (
            f"Total: {s.total_input_tokens:,}in/{s.total_output_tokens:,}out, "
            f"${s.total_cost_usd:.4f}"
        )
        return " | ".join(parts) + f" | {total}"
