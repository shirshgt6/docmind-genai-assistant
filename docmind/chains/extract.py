"""Structured extraction: document text -> validated Pydantic object."""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from docmind.chains.structured import structured_chain
from docmind.prompts import EXTRACT_PROMPT


class DocumentInsights(BaseModel):
    title: str = Field(description="A short title for the document")
    document_type: Literal["resume", "invoice", "contract", "policy", "report", "article", "other"] = Field(
        description="What kind of document this is"
    )
    key_points: list[str] = Field(description="3-6 most important points")
    entities: list[str] = Field(default_factory=list, description="People, organisations, products, places")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall tone")


def build_extract_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"text": str}. Output: DocumentInsights."""
    return structured_chain(llm, EXTRACT_PROMPT, DocumentInsights)
