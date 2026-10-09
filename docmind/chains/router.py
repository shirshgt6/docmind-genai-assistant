"""Intent router: classify a request with structured output (the service dispatches on it)."""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from docmind.chains.structured import structured_chain
from docmind.prompts import ROUTER_PROMPT


class Intent(BaseModel):
    intent: Literal["qa", "summarize", "extract"] = Field(description="Which pipeline should handle the request")


def build_router_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"question": str}. Output: Intent."""
    return structured_chain(llm, ROUTER_PROMPT, Intent)
