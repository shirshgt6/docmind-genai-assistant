"""Structured output that works on any chat model.

Models with native tool/JSON support use with_structured_output(); others fall
back to a PydanticOutputParser with format instructions injected into the prompt.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel


def structured_chain(llm: BaseChatModel, prompt: ChatPromptTemplate, schema: type[BaseModel]) -> Runnable:
    try:
        return prompt | llm.with_structured_output(schema)
    except NotImplementedError:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt_with_format = prompt + [("human", "Respond with JSON only.\n{format_instructions}")]
        return prompt_with_format.partial(format_instructions=parser.get_format_instructions()) | llm | parser
