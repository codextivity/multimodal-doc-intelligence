# src/test_multimodal_chain.py
# Tests the multimodal RAG chain with questions about
# both text and visual content.

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from multimodal_ingestion import ingest_to_vectorstore, load_vectorstore
from multimodal_chain import build_multimodal_rag_chain

def test_chain():
    """
    Tests the multimodal chain with questions targeting
    different content types.
    """
    # Load existing vector store or ingest test files
    chroma_path = Path("chroma_db")

    if chroma_path.exists():
        print("Loading existing vector store...")
        vectorstore = load_vectorstore()
    else:
        print("No vector store found — ingest some documents first")
        return

    # Show what is in the vector store
    all_docs = vectorstore.get()
    content_types = {}
    for meta in all_docs["metadatas"]:
        ct = meta.get("content_type", "unknown")
        content_types[ct] = content_types.get(ct, 0) + 1

    print(f"\nVector store contents:")
    for ct, count in content_types.items():
        print(f"  {ct}: {count} chunks")

    # Build chain
    chain = build_multimodal_rag_chain(vectorstore)

    # Test questions targeting different content types
    test_questions = [
        # Targets image chunk (eye colour chart)
        "What does the bar chart show about eye colours?",

        # Targets text chunk (Cambodia PDF)
        "What was Cambodia's GDP in 2013?",

        # Should retrieve both types
        "What visual content is available in the documents?",

        # Follow-up test for history-aware retrieval
    ]

    history = []

    print("\n" + "=" * 60)
    print("MULTIMODAL RAG CHAIN TEST")
    print("=" * 60)

    for question in test_questions:
        print(f"\nYou: {question}")

        answer = chain.invoke({
            "input": question,
            "chat_history": history
        })

        print(f"Assistant: {answer}")

        # Update history for follow-up questions
        from langchain_core.messages import HumanMessage, AIMessage
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

    # Test follow-up question
    print(f"\nYou: Which year group had more blue-eyed students?")
    answer = chain.invoke({
        "input": "Which year group had more blue-eyed students?",
        "chat_history": history
    })
    print(f"Assistant: {answer}")

if __name__ == "__main__":
    test_chain()