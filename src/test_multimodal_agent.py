# src/test_multimodal_agent.py
# Add project root to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.multimodal_ingestion import load_vectorstore
from app.core.multimodal_agent import build_multimodal_agent, run_multimodal_agent

def test_agent():
    vectorstore = load_vectorstore()
    agent = build_multimodal_agent(vectorstore)

    test_questions = [
        # Routes to text node
        ("text", "What was Cambodia's GDP in 2013?"),

        # Routes to vision node
        ("vision", "What does the bar chart about eye colours show?"),

        # Routes to tools node
        ("tools", "Calculate the CAGR from 2013 to 2023 using GDP of 15228 and 31940"),

        # Routes to compare node
        ("compare", "How does the visual data compare to the text data in the documents?"),

        # Routes to web node
        ("web", "What is Cambodia's current GDP in 2025?"),
    ]

    print("MULTIMODAL AGENT ROUTING TEST")
    print("=" * 60)

    for expected_route, question in test_questions:
        print(f"\nExpected route: {expected_route}")
        print(f"Question: {question}")
        print("-" * 40)
        answer = run_multimodal_agent(agent, question)
        print(f"Answer: {answer[:300]}...")

if __name__ == "__main__":
    test_agent()