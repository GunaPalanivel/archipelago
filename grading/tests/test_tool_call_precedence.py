"""Tests for the TOOL_CALL_PRECEDENCE verifier."""

from typing import Any

import pytest

from runner.evals.models import EvalConfig, EvalIds, EvalImplInput
from runner.evals.tool_call_precedence.main import tool_call_precedence_eval
from runner.models import (
    AgentStatus,
    AgentTrajectoryOutput,
    GradingSettings,
    Verifier,
    VerifierResultStatus,
)


def _assistant(*calls: tuple[str, str, str]) -> dict[str, Any]:
    """An assistant turn issuing tool calls, each given as (id, name, args_json)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            for cid, name, args in calls
        ],
    }


def _tool_result(call_id: str, content: str = "ok") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _make_input(
    messages: list[dict[str, Any]], verifier_values: dict[str, Any]
) -> EvalImplInput:
    return EvalImplInput(
        initial_snapshot_bytes=None,
        final_snapshot_bytes=None,
        trajectory=AgentTrajectoryOutput(
            messages=messages,
            status=AgentStatus.COMPLETED,
            time_elapsed=1.0,
        ),
        grading_settings=GradingSettings(llm_judge_model="openai/gpt-4o-mini"),
        verifier=Verifier(
            verifier_id="tool_call_precedence",
            verifier_version=1,
            world_id=None,
            task_id=None,
            eval_config_id="cfg",
            verifier_values=verifier_values,
            verifier_index=0,
        ),
        eval_config=EvalConfig(
            eval_config_id="cfg",
            eval_config_name="cfg",
            eval_defn_id=EvalIds.TOOL_CALL_PRECEDENCE,
            eval_config_values={},
        ),
        dependencies=None,
        helper_results=None,
    )


async def _run(messages: list[dict[str, Any]], **verifier_values: Any) -> Any:
    return await tool_call_precedence_eval(_make_input(messages, verifier_values))


# ---------------------------------------------------------------------------
# Core ordering
# ---------------------------------------------------------------------------


async def test_passes_when_before_precedes_after():
    messages = [
        {"role": "user", "content": "build the deliverable"},
        _assistant(("c1", "search_mail", "{}")),
        _tool_result("c1"),
        _assistant(("c2", "write_spreadsheet", "{}")),
        _tool_result("c2"),
    ]
    result = await _run(
        messages, before_tool="search_mail", after_tool="write_spreadsheet"
    )
    assert result.score == 1.0
    assert result.status == VerifierResultStatus.OK
    assert result.verifier_result_values["before_first_call_number"] == 1
    assert result.verifier_result_values["after_first_call_number"] == 2


async def test_fails_when_after_precedes_before():
    """The agent wrote first and only then read. Two TOOL_CALL_CHECKs would
    both pass here, which is exactly the blind spot this verifier covers."""
    messages = [
        {"role": "user", "content": "build the deliverable"},
        _assistant(("c1", "write_spreadsheet", "{}")),
        _tool_result("c1"),
        _assistant(("c2", "search_mail", "{}")),
        _tool_result("c2"),
    ]
    result = await _run(
        messages, before_tool="search_mail", after_tool="write_spreadsheet"
    )
    assert result.score == 0.0
    assert result.verifier_result_values["before_first_call_number"] == 2
    assert result.verifier_result_values["after_first_call_number"] == 1


async def test_fails_when_before_never_called():
    messages = [
        {"role": "user", "content": "build the deliverable"},
        _assistant(("c1", "write_spreadsheet", "{}")),
        _tool_result("c1"),
    ]
    result = await _run(
        messages, before_tool="search_mail", after_tool="write_spreadsheet"
    )
    assert result.score == 0.0
    assert result.verifier_result_values["before_found"] is False
    assert "without any preceding" in result.message


async def test_uses_earliest_after_call_not_the_last():
    """A late compliant read must not excuse an early uninformed write."""
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "write_spreadsheet", "{}")),
        _tool_result("c1"),
        _assistant(("c2", "search_mail", "{}")),
        _tool_result("c2"),
        _assistant(("c3", "write_spreadsheet", "{}")),
        _tool_result("c3"),
    ]
    result = await _run(
        messages, before_tool="search_mail", after_tool="write_spreadsheet"
    )
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Missing committing action
# ---------------------------------------------------------------------------


async def test_after_missing_fails_when_required():
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "search_mail", "{}")),
        _tool_result("c1"),
    ]
    result = await _run(
        messages,
        before_tool="search_mail",
        after_tool="write_spreadsheet",
        require_after_call=True,
    )
    assert result.score == 0.0
    assert "never called" in result.message


async def test_after_missing_passes_when_not_required():
    """Pure 'never act without checking first' guard: no action, no violation."""
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "search_mail", "{}")),
        _tool_result("c1"),
    ]
    result = await _run(
        messages,
        before_tool="search_mail",
        after_tool="delete_file",
        require_after_call=False,
    )
    assert result.score == 1.0


async def test_empty_trajectory_fails_when_after_required():
    result = await _run(
        [{"role": "user", "content": "go"}],
        before_tool="search_mail",
        after_tool="write_spreadsheet",
    )
    assert result.score == 0.0
    assert result.verifier_result_values["total_tool_calls"] == 0


# ---------------------------------------------------------------------------
# Name lists and argument filtering
# ---------------------------------------------------------------------------


async def test_accepts_list_of_before_tool_names():
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "read_mail", "{}")),
        _tool_result("c1"),
        _assistant(("c2", "write_spreadsheet", "{}")),
        _tool_result("c2"),
    ]
    result = await _run(
        messages,
        before_tool=["search_mail", "read_mail"],
        after_tool="write_spreadsheet",
    )
    assert result.score == 1.0


async def test_before_args_filter_excludes_non_matching_calls():
    """Reading the wrong mailbox is not reading the one that carries the update."""
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "search_mail", '{"folder": "spam"}')),
        _tool_result("c1"),
        _assistant(("c2", "write_spreadsheet", "{}")),
        _tool_result("c2"),
    ]
    result = await _run(
        messages,
        before_tool="search_mail",
        after_tool="write_spreadsheet",
        before_args={"folder": "inbox"},
    )
    assert result.score == 0.0
    assert result.verifier_result_values["before_found"] is False


async def test_before_args_regex_mode():
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "search_mail", '{"query": "revised discount rate"}')),
        _tool_result("c1"),
        _assistant(("c2", "write_spreadsheet", "{}")),
        _tool_result("c2"),
    ]
    result = await _run(
        messages,
        before_tool="search_mail",
        after_tool="write_spreadsheet",
        before_args={"query": r"discount\s+rate"},
        expected_args_regex=True,
    )
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Gateway CLI ordering — regression guard
# ---------------------------------------------------------------------------


async def test_gateway_wrapped_call_is_ordered_by_call_number():
    """Gateway-synthesized calls are appended, not interleaved, by the extractor.

    Here the agent searches mail through the mcp_cli gateway at step 1, writes
    at step 2, then searches mail natively again at step 3. The concatenation
    ``native + unwrap_gateway_cli_calls(native)`` yields
    ``[run_command(1), write_spreadsheet(2), search_mail(3), search_mail(1)]``,
    so scanning in list order finds the step-3 read first and wrongly concludes
    the write came earlier. Sorting by call_number is what makes this pass.
    """
    gateway_cmd = (
        '{"command": "mcp_cli --method=execute --connector=mail '
        '--tool=search_mail --params={\\"query\\": \\"rate\\"}"}'
    )
    messages = [
        {"role": "user", "content": "go"},
        _assistant(("c1", "run_command", gateway_cmd)),
        _tool_result("c1"),
        _assistant(("c2", "write_spreadsheet", "{}")),
        _tool_result("c2"),
        _assistant(("c3", "search_mail", "{}")),
        _tool_result("c3"),
    ]
    result = await _run(
        messages, before_tool="search_mail", after_tool="write_spreadsheet"
    )
    assert result.verifier_result_values["before_first_call_number"] == 1
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"after_tool": "write_spreadsheet"},
        {"before_tool": "search_mail"},
        {"before_tool": "", "after_tool": "write_spreadsheet"},
        {"before_tool": "search_mail", "after_tool": []},
    ],
)
async def test_missing_config_returns_error(values: dict[str, Any]):
    result = await _run([{"role": "user", "content": "go"}], **values)
    assert result.status == VerifierResultStatus.ERROR
    assert result.score == 0.0
