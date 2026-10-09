from langchain_core.messages import AIMessage

from docmind.agent import build_agent
from docmind.tools import build_tools, calculator, current_date
from tests.conftest import FakeToolChatModel


def test_calculator_evaluates_safely():
    assert calculator.invoke({"expression": "(1200 * 12) * 0.18"}) == "2592.0"
    assert calculator.invoke({"expression": "10 / 0"}).startswith("Error")
    assert calculator.invoke({"expression": "__import__('os').system('ls')"}).startswith("Error")
    assert calculator.invoke({"expression": "9 ** 999999"}).startswith("Error")


def test_current_date_is_iso():
    assert len(current_date.invoke({})) == 10


def test_search_tool_only_added_when_documents_exist():
    assert [t.name for t in build_tools(None)] == ["calculator", "current_date"]


def test_agent_calls_tool_then_answers():
    scripted = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "24 - 10"}, "id": "c1"}]),
            AIMessage(content="You have 14 days left."),
        ]
    )
    executor = build_agent(FakeToolChatModel(messages=scripted), retriever=None)

    result = executor.invoke({"input": "I used 10 of 24 leave days, how many left?"})

    assert result["output"] == "You have 14 days left."
    action, observation = result["intermediate_steps"][0]
    assert action.tool == "calculator"
    assert observation == "14"
