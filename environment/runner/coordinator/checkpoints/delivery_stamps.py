"""
A delivery stamp is the server's own record of a tool call: which channel it
landed on and which actors it actually reached. Apps return stamps in the tool
result's structured content; the coordinator reads them back off the recorded
observation so events can key on resolved recipients instead of argument text.
"""

from collections.abc import Iterable

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from ..events.models import (
    ChannelCondition,
    DeliveryStampCondition,
    ToolCallCondition,
)
from ..utils import (
    HAS_STRUCTURED_CONTENT_SUMMARY_KEY,
    STRUCTURED_CONTENT_SUMMARY_KEY,
)
from .models import ToolCallCheckpointObservation

DELIVERY_STAMPS_RESULT_KEY = "delivery_stamps"

RecognizedToolCallCondition = ChannelCondition | DeliveryStampCondition


class DeliveryStamp(BaseModel):
    channel_key: str | None = None
    actor_ids: list[str] = Field(default_factory=list)


class ObservedDelivery(BaseModel):
    # An app can only stamp a call through structured content, so its absence is
    # what marks a call the server never resolved. Stamps can be empty, malformed
    # or dropped for size; each of those is a resolution that reached nobody.
    resolved: bool = False
    stamps: list[DeliveryStamp] = Field(default_factory=list)


def read_observed_delivery(
    tool_call: ToolCallCheckpointObservation,
) -> ObservedDelivery:
    result_summary = tool_call.result_summary or {}
    resolved = result_summary.get(HAS_STRUCTURED_CONTENT_SUMMARY_KEY) is True
    if not resolved:
        return ObservedDelivery()
    structured_content = result_summary.get(STRUCTURED_CONTENT_SUMMARY_KEY)
    if not isinstance(structured_content, dict):
        return ObservedDelivery(resolved=True)
    raw_stamps = structured_content.get(DELIVERY_STAMPS_RESULT_KEY)
    if not isinstance(raw_stamps, list):
        return ObservedDelivery(resolved=True)
    stamps: list[DeliveryStamp] = []
    for raw_stamp in raw_stamps:
        try:
            stamps.append(DeliveryStamp.model_validate(raw_stamp))
        except ValidationError as error:
            logger.warning(
                "Environment Coordinator ignoring malformed delivery stamp "
                + f"sequence={tool_call.sequence} tool={tool_call.tool_name} "
                + f"error={error!r}"
            )
    return ObservedDelivery(resolved=True, stamps=stamps)


def recognized_conditions(
    conditions: Iterable[ToolCallCondition],
) -> list[RecognizedToolCallCondition]:
    # A condition this image cannot read is skipped rather than refused, and a
    # selector left with none falls back to its argument conditions.
    return [
        condition
        for condition in conditions
        if isinstance(condition, ChannelCondition | DeliveryStampCondition)
    ]


def delivery_satisfies(
    conditions: Iterable[RecognizedToolCallCondition], delivery: ObservedDelivery
) -> bool:
    # One delivery has to satisfy every condition. Spreading them across separate
    # stamps would fire on a channel and an actor that never met.
    return any(
        all(_condition_matches(condition, stamp) for condition in conditions)
        for stamp in delivery.stamps
    )


def _condition_matches(
    condition: RecognizedToolCallCondition, stamp: DeliveryStamp
) -> bool:
    if isinstance(condition, ChannelCondition):
        return stamp.channel_key == condition.channel_key
    if isinstance(condition, DeliveryStampCondition):
        return bool(set(condition.actor_ids).intersection(stamp.actor_ids))
    raise ValueError(f"Unknown ToolCallCondition: {condition}")
