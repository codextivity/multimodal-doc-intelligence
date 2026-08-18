# 🧠 Multimodal Document Intelligence Copilot

AI research assistant that understands text, charts, tables, and scanned documents. Upload any document type and ask questions across all content modalities.

**Live Demo:** https://your-render-url.onrender.com/docs **GitHub:** https://github.com/codextivity/multimodal-doc-intelligence

---

## What Makes This Different

Most RAG systems only handle text. This system handles:

| Content Type | How It Works | Example Query |
| --- | --- | --- |
| Text PDF | PyMuPDF text extraction | "What was GDP in 2013?" |
| Charts/Graphs | GPT-4o vision description | "What does the bar chart show?" |
| Scanned PDFs | VLM page-by-page analysis | "Extract text from this scan" |
| Mixed PDFs | Text + VLM combined | "Summarize page 3 including charts" |
| Images | Direct VLM processing | "What data is in this image?" |

---

## Agent Routing

The LangGraph agent has 5 specialized nodes:

```
Question
    │
    ▼
Supervisor (classifies intent)
    │
    ├──► TextNode    "What was GDP in 2013?"
    ├──► VisionNode  "What does the chart show?"
    ├──► ToolsNode   "Calculate the CAGR"
    ├──► CompareNode "Compare chart vs text data"
    └──► WebNode     "What is current GDP?"
```

Each node uses the optimal retrieval strategy for its content type.

---

## Tech Stack

| Component | Technology |
| --- | --- |
| Vision LLM | GPT-4o multimodal API |
| Text LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector store | ChromaDB |
| Agent | LangGraph 5-node supervisor |
| Framework | LangChain 1.3 |
| API | FastAPI |
| Observability | LangSmith |
| Image processing | Pillow + PyMuPDF |
| Web search | Tavily |
| Deployment | Render + Docker |
