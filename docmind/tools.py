"""Tools the agent may call. Each one is small, deterministic and safe."""
import ast
import operator
from datetime import date

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from docmind.chains.rag import format_docs

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(1200 * 12) * 0.18'. Use for ALL math."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: {exc}"
    return str(round(result, 6))


@tool
def current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


def make_search_tool(retriever: BaseRetriever) -> BaseTool:
    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents and return the most relevant passages with source ids."""
        docs = retriever.invoke(query)
        return format_docs(docs) if docs else "No relevant passages found."

    return search_documents


def build_tools(retriever: BaseRetriever | None) -> list[BaseTool]:
    tools: list[BaseTool] = [calculator, current_date]
    if retriever is not None:
        tools.insert(0, make_search_tool(retriever))
    return tools
