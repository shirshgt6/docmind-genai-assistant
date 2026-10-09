"""All prompt templates in one place (ChatPromptTemplate + MessagesPlaceholder)."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Rewrites a follow-up ("what about its price?") into a standalone question using chat history.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the question so it can be "
            "understood without the history. Do NOT answer it. Return only the rewritten question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocMind, an assistant that answers ONLY from the provided context.\n"
            "Rules:\n"
            "- If the answer is not in the context, say: \"I couldn't find that in the uploaded documents.\"\n"
            "- Cite the sources you used like [source#chunk].\n"
            "- Be concise and factual.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ]
)

SUMMARY_STUFF_PROMPT = ChatPromptTemplate.from_template(
    "Write a clear summary of the following document in {style} style.\n\n{text}"
)

SUMMARY_MAP_PROMPT = ChatPromptTemplate.from_template(
    "Summarize this part of a larger document in 3-4 sentences, keeping key facts and numbers:\n\n{text}"
)

SUMMARY_REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "These are summaries of consecutive parts of one document. Combine them into one coherent "
    "summary in {style} style, without repeating points.\n\n{text}"
)

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract structured information from the document. Use only facts present in the text."),
        ("human", "{text}"),
    ]
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's request about their uploaded documents into exactly one intent:\n"
            "- qa: a specific question that should be answered from the documents\n"
            "- summarize: asks for a summary / overview / TL;DR\n"
            "- extract: asks for structured details (key points, entities, document type, sentiment)",
        ),
        ("human", "{question}"),
    ]
)

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocMind Agent. Use tools when they help:\n"
            "- search_documents for anything about the user's uploaded documents\n"
            "- calculator for any arithmetic (never do math in your head)\n"
            "- current_date for questions about today's date or deadlines\n"
            "Answer the user directly once you have what you need.",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
