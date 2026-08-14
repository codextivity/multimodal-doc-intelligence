# src/test_ingestion.py
# Tests the complete multimodal ingestion pipeline.

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from document_detector import classify_document
from multimodal_ingestion import ingest_document, ingest_to_vectorstore

def test_classification():
    """Test document classification on your test files."""
    print("=" * 60)
    print("DOCUMENT CLASSIFICATION TEST")
    print("=" * 60)

    # Test your existing test images
    test_files = list(Path("../test_images").glob("*.png")) + \
                 list(Path("../test_images").glob("*.jpg"))

    # Add test PDFs if you have any
    test_files += list(Path("../test_docs").glob("*.pdf"))

    for file_path in test_files:
        doc_type, metadata = classify_document(str(file_path))
        print(f"\nFile: {file_path.name}")
        print(f"Type: {doc_type.value}")
        print(f"Info: {metadata}")

def test_full_ingestion():
    """Test complete ingestion on one image file."""
    print("\n" + "=" * 60)
    print("FULL INGESTION TEST")
    print("=" * 60)

    # Use your chart image from test_images/
    test_images = list(Path("../test_images").glob("*.png")) + \
                  list(Path("../test_images").glob("*.jpg"))

    if not test_images:
        print("No test images found")
        return

    # Ingest first image
    test_file = str(test_images[0])
    print(f"Ingesting: {Path(test_file).name}")

    vectorstore = ingest_to_vectorstore(test_file)

    # Test retrieval
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    test_queries = [
        "eye colour frequency",
        "bar chart data",
        "Year 7 students"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            print(f"  Result {i+1}: [{doc.metadata['content_type']}] "
                  f"page {doc.metadata['page']} | "
                  f"{doc.page_content[:100]}...")

if __name__ == "__main__":
    test_classification()
    test_full_ingestion()