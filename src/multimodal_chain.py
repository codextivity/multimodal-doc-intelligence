# src/multimodal_chain.py
# Conversational RAG chain with multimodal source awareness.
#
# Key difference from LangChain Copilot chain.py:
# Every retrieved chunk has a content_type in its metadata.
# We use this to format context differently and generate
# citations that tell the user whether the answer came from
# text, a chart, a table image, or a scanned page.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Content type labels ───────────────────────────────────────────────────────
# Human-readable labels for each content type.
# These appear in citations so users know what kind of source was used.

CONTENT_TYPE_LABELS = {
    "text":    "📄 Text",
    "image":   "🖼️ Image",
    "chart":   "📊 Chart",
    "scanned": "📷 Scanned page",
    "mixed":   "📑 Mixed content",
}

# ── Question rewriter prompt ─────────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and the user's latest question,
rewrite the question as a complete standalone question.

Rules:
- Replace all pronouns with their actual referents
- Replace relative references with explicit ones
- If the question asks about a chart or visual, keep that context
- Return ONLY the rewritten question"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ── Answer generation prompt ─────────────────────────────────────────────────
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a multimodal document intelligence assistant.
You can answer questions about both text documents and visual content
including charts, tables, diagrams, and scanned documents.

Answer the question using the context below.
The context includes content from different source types — text, charts,
images, and scanned pages. Each source is labeled with its type and page.

Rules:
- Always cite your sources using the format [Source type, Page X]
- If the answer comes from a chart, mention what the chart shows
- If the answer comes from a scanned page, note it may contain OCR artifacts
- If the answer comes from multiple source types, cite each one
- If the information is not in the context, say so clearly
- Be specific with numbers and data points from visual sources

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def format_multimodal_docs(docs: list[Document]) -> str:
    """
    Formats retrieved documents into context string with type-aware labels.

    Unlike the original format_docs, this version:
    1. Labels each chunk with its content type
    2. Groups chunks by type for clarity
    3. Adds special formatting for visual content descriptions

    The LLM uses these labels to generate type-specific citations.
    """
    if not docs:
        return "No relevant content found in the documents."

    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        content_type = doc.metadata.get("content_type", "text")
        file_name = doc.metadata.get("file_name", "unknown")

        # Get human-readable label for this content type
        type_label = CONTENT_TYPE_LABELS.get(content_type, "📄 Content")

        # Format differently based on content type
        if content_type in ("chart", "image", "scanned", "mixed"):
            # Visual content — add a note that this is a description
            header = (f"[Source {i+1} | {type_label} | "
                     f"File: {file_name} | Page: {page}]\n"
                     f"[This is a description of visual content]")
        else:
            # Text content — standard format
            header = (f"[Source {i+1} | {type_label} | "
                     f"File: {file_name} | Page: {page}]")

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)

def build_multimodal_rag_chain(vectorstore):
    """
    Builds a conversational RAG chain with multimodal source awareness.

    The chain works identically to the LangChain Copilot chain but with
    two key additions:
    1. format_multimodal_docs adds content-type labels to context
    2. ANSWER_PROMPT instructs the LLM to cite source types

    Args:
        vectorstore: ChromaDB vector store with multimodal chunks

    Returns:
        A chain that takes {"input": str, "chat_history": list}
        and returns a cited answer string
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Use k=6 to retrieve more chunks
    # With mixed content types, we want enough chunks to cover
    # both text and visual sources for each query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # Question rewriter — same pattern as LangChain Copilot
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    # History-aware retriever
    history_aware_retriever = RunnableBranch(
        (
            lambda x: bool(x.get("chat_history")),
            rewrite_chain | retriever
        ),
        RunnableLambda(lambda x: x["input"]) | retriever
    )

    # Full RAG chain with multimodal formatting
    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_multimodal_docs(
                    history_aware_retriever.invoke(x)
                )
            )
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain