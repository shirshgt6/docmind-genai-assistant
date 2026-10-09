"""Tool-calling agent (LLM decides which tool to call, AgentExecutor runs the loop)."""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docmind.prompts import AGENT_PROMPT
from docmind.tools import build_tools


def build_agent(llm: BaseChatModel, retriever: BaseRetriever | None) -> AgentExecutor:
    tools = build_tools(retriever)
    agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,  # hard stop: an agent must never loop forever on our bill
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
