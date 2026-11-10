# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project observed Responses exchanges without claiming unobserved tool execution."""

from __future__ import annotations

import json
from pathlib import Path

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory import AgentInvocation, ModelCallRef, ObservationGap, TrajectoryRecord, TrajectoryTurn
from responses_api_agents.ember_bridge_agent.proxy import ModelProxy


def observed_trajectory(
    proxy: ModelProxy | None, *, model_ref: ModelServerRef, rollout_id: str, output: Path
) -> TrajectoryRecord:
    """Keep real model turns and owner-reported completion; grade never sets agent status."""
    turns = []
    refs = []
    gaps = []
    status = "unknown"
    try:
        receipt = json.loads((output / "native/actor/receipt.json").read_text())
        if isinstance(receipt, dict) and receipt.get("finished") is True:
            status = "completed"
        elif isinstance(receipt, dict) and receipt.get("finished") is False:
            status = "incomplete"
    except (OSError, ValueError):
        pass
    if status == "unknown":
        gaps.append(ObservationGap(code="invocation_status_unavailable", invocation_id="root"))
    for step, exchange in enumerate(proxy.exchanges if proxy else [], start=1):
        response = exchange.response or {}
        response_id = response.get("id")
        call_refs = []
        if isinstance(response_id, str) and response_id:
            call_refs.append(ModelCallRef(model_ref=model_ref, response_id=response_id))
            refs.extend(call_refs)
        else:
            gaps.append(
                ObservationGap(code="model_call_reference_unavailable", invocation_id="root", detail=f"turn:{step}")
            )
        output_items = response.get("output")
        items = output_items if isinstance(output_items, list) else []
        if any(isinstance(item, dict) and item.get("type") == "function_call" for item in items):
            gaps.append(
                ObservationGap(
                    code="tool_execution_unavailable",
                    invocation_id="root",
                    detail=f"turn:{step}: proxy observes tool requests, not execution or final tool results",
                )
            )
        if exchange.request is None or exchange.response is None:
            gaps.append(ObservationGap(code="model_exchange_incomplete", invocation_id="root", detail=f"turn:{step}"))
        turns.append(
            TrajectoryTurn(
                invocation_id="root",
                task_id="unscoped",
                rollout_id=rollout_id,
                turn_no=step,
                timestamp=exchange.started_at,
                question=exchange.request.get("input") if exchange.request else None,
                answer=[item for item in items if not isinstance(item, dict) or item.get("type") != "reasoning"],
                reasoning_content=[
                    item for item in items if isinstance(item, dict) and item.get("type") == "reasoning"
                ]
                or None,
                step_count=step,
                model_calls=call_refs,
            )
        )
    if not turns:
        gaps.append(ObservationGap(code="model_exchange_unavailable", invocation_id="root"))
    return TrajectoryRecord(
        task_id="unscoped",
        rollout_id=rollout_id,
        invocations=[AgentInvocation(invocation_id="root", status=status, model_calls=refs)],
        turns=turns,
        gaps=gaps,
    )
