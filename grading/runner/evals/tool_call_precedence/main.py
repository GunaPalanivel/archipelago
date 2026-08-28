"""Tool Call Precedence eval - deterministic check that one tool precedes another.

TOOL_CALL_CHECK answers "was this tool called?". It filters the trajectory by
tool name and checks each match independently, so it cannot express an ordering
constraint between two different tools. The trajectory already carries the
information — ``extract_tool_calls_with_outputs`` attaches a sequential
``call_number`` to every call — it is simply never compared across tools.

This verifier closes that gap. It asserts that one tool was called BEFORE
another, which is what "check the state before you change it" means in practice:
read the config before editing it, or search the mail thread before writing the
deliverable.

An agent that edits a file and only afterwards reads what it clobbered satisfies
two TOOL_CALL_CHECKs and still did the wrong thing.

Configuration via verifier_values:
    before_tool (str | list[str], required): Tool that must come first. A list
        matches if the call's name is any of them.
    after_tool (str | list[str], required): Tool whose FIRST occurrence must be
        preceded by ``before_tool``.
    before_args (dict, optional): Arguments the "before" call must match for it
        to count. Only matching calls are considered.
    after_args (dict, optional): Arguments the "after" call must match for it to
        count.
    require_after_call (bool, optional): If true (default), FAIL when the
        "after" tool was never called — the committing action is expected to
        have happened. If false, PASS vacuously in that case, which is the
        right reading for a pure "never act without checking first" guard.
    expected_args_contains (bool, optional): Substring match for string values
        in before_args/after_args. Ignored when expected_args_regex is true.
    expected_args_regex (bool, optional): Match string values with re.search().
        Takes precedence over expected_args_contains.

Example verifier_values:
    {
        "before_tool": ["search_mail", "read_mail"],
        "after_tool": "write_spreadsheet",
        "require_after_call": true
    }
"""

from typing import Any

from loguru import logger

from runner.evals.models import EvalImplInput

# Reuse tool_call_check's argument matching rather than restating it. Two copies
# would drift, and a precedence check that matched arguments differently from
# TOOL_CALL_CHECK on the same task would be a quiet trap.
from runner.evals.tool_call_check.main import (
    _args_match,
    _parse_arguments,
    _resolve_string_match_mode,
)
from runner.models import VerifierResult, VerifierResultStatus
from runner.utils.trajectory import (
    extract_tool_calls_with_outputs,
    unwrap_gateway_cli_calls,
)


def _name_set(raw: Any) -> set[str]:
    """Accept a single tool name or a list of acceptable names."""
    if isinstance(raw, str):
        return {raw} if raw else set()
    if isinstance(raw, (list, tuple)):
        return {t for t in raw if isinstance(t, str) and t}
    return set()


def _display_name(raw: Any, names: set[str]) -> str:
    return raw if isinstance(raw, str) else ", ".join(sorted(names))


def _first_matching_call(
    tool_calls: list[dict[str, Any]],
    names: set[str],
    expected_args: dict[str, Any] | None,
    string_match_mode: str,
) -> dict[str, Any] | None:
    """Earliest call whose name is in ``names`` and whose args match, if any.

    ``tool_calls`` must already be ordered by ``call_number``.
    """
    for tc in tool_calls:
        if tc["tool_name"] not in names:
            continue
        if expected_args and not _args_match(
            _parse_arguments(tc["arguments"]),
            expected_args,
            string_match_mode=string_match_mode,
        ):
            continue
        return tc
    return None


def _ordered_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """All tool calls, including gateway-wrapped ones, in execution order.

    ``unwrap_gateway_cli_calls`` returns synthesized entries for MCP tools
    invoked through the ``mcp_cli`` shell gateway, and callers concatenate them
    onto the native list. That concatenation is not in ``call_number`` order:
    every synthesized call lands after every native one. An ordering check has
    to sort before it can compare, which is why this helper exists.

    A synthesized call inherits the ``call_number`` of the shell command that
    wrapped it, so ties with the wrapper are expected and the comparison is
    strict.
    """
    native = extract_tool_calls_with_outputs(messages)
    return sorted(
        native + unwrap_gateway_cli_calls(native), key=lambda tc: tc["call_number"]
    )


async def tool_call_precedence_eval(input: EvalImplInput) -> VerifierResult:
    """
    Deterministic verifier that checks one tool was called before another.

    Passes when the first argument-matching call to ``before_tool`` has a
    strictly lower ``call_number`` than the first argument-matching call to
    ``after_tool``.

    Returns:
        VerifierResult with:
        - score: 1.0 if the precedence holds, 0.0 otherwise
        - verifier_result_values containing both call numbers and the reason
    """
    verifier_id = input.verifier.verifier_id
    verifier_version = input.verifier.verifier_version
    verifier_values = input.verifier.verifier_values or {}

    raw_before = verifier_values.get("before_tool")
    raw_after = verifier_values.get("after_tool")
    before_args = verifier_values.get("before_args")
    after_args = verifier_values.get("after_args")
    require_after_call = bool(verifier_values.get("require_after_call", True))
    string_match_mode = _resolve_string_match_mode(verifier_values)

    before_names = _name_set(raw_before)
    after_names = _name_set(raw_after)

    if not before_names or not after_names:
        return VerifierResult(
            verifier_id=verifier_id,
            verifier_version=verifier_version,
            score=0.0,
            status=VerifierResultStatus.ERROR,
            verifier_result_values={
                "error": "before_tool and after_tool are both required (string or list of strings)"
            },
            message="Configuration error: before_tool and after_tool are required",
        )

    before_label = _display_name(raw_before, before_names)
    after_label = _display_name(raw_after, after_names)

    logger.info(f"Checking precedence: '{before_label}' must precede '{after_label}'")

    tool_calls = _ordered_calls(input.trajectory.messages)

    before_call = _first_matching_call(
        tool_calls, before_names, before_args, string_match_mode
    )
    after_call = _first_matching_call(
        tool_calls, after_names, after_args, string_match_mode
    )

    before_n = before_call["call_number"] if before_call else None
    after_n = after_call["call_number"] if after_call else None

    if after_call is None:
        # Nothing was committed. Whether that is a pass depends on what the task
        # asked for, so it is the caller's call rather than ours.
        passed = not require_after_call
        message = f"'{after_label}' was never called" + (
            "" if passed else " but was required"
        )
    elif before_call is None:
        passed = False
        message = (
            f"'{after_label}' was called at step {after_n} without any preceding "
            f"'{before_label}'"
        )
    else:
        passed = before_n < after_n
        message = (
            f"'{before_label}' at step {before_n} precedes '{after_label}' at step {after_n}"
            if passed
            else f"'{before_label}' at step {before_n} does not precede "
            f"'{after_label}' at step {after_n}"
        )

    logger.info(f"Tool call precedence result: {message}")

    return VerifierResult(
        verifier_id=verifier_id,
        verifier_version=verifier_version,
        score=1.0 if passed else 0.0,
        status=VerifierResultStatus.OK,
        verifier_result_values={
            "before_tool": before_label,
            "after_tool": after_label,
            "before_first_call_number": before_n,
            "after_first_call_number": after_n,
            "before_found": before_call is not None,
            "after_found": after_call is not None,
            "require_after_call": require_after_call,
            "before_args": before_args,
            "after_args": after_args,
            "expected_args_string_match_mode": string_match_mode,
            "total_tool_calls": len(tool_calls),
            "message": message,
        },
        message=message,
    )
