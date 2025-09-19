# Standard RAG Text

This project implements different **text chunking strategies** for use in **RAG (Retrieval-Augmented Generation)** pipelines.  
The FastAPI server provides endpoints to process text into smaller chunks, which can then be indexed or embedded for retrieval.

---

## 🚀 Features

- **Fixed Chunking** → Splits text into fixed-size chunks.  
- **Paragraph Chunking** → Splits text into paragraphs while preserving structure.  
- **Sentence Chunking** → Splits text into sentences (using NLP parsing).  
- **Sliding Window Chunking** → Creates overlapping chunks for context continuity.  
- **Hybrid Chunking** → Combines paragraph boundaries with sliding window overlap.  

---

## 📂 Project Structure

```
standard-rag-text/
│── main.py              # FastAPI entrypoint
│── requirements.txt     # Dependencies
│── .gitignore
│── __init__.py
│
│── app/
│   ├── config.py        # Configurations
│   ├── constants.py     # Constant values
│   ├── schema.py        # Pydantic schemas for API requests
│
│── data/
│   ├── alice.txt        # Sample text file
│   ├── sherlock.txt     # Sample text file
│
│── text_chunk/
│   ├── fixed.py         # Fixed-size chunking logic
│   ├── paragraph.py     # Paragraph-based chunking logic
│   ├── sentance.py      # Sentence-based chunking logic
│   ├── sliding_window.py# Sliding window chunking
│   ├── hybrid.py        # Hybrid (paragraph + sliding window) chunking
│   ├── __init__.py
│
│── utils/
│   ├── logger.py        # Centralized logger
│   ├── __init__.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-repo>/standard-rag-text.git
cd standard-rag-text
pip install -r requirements.txt
```

---

## ▶️ Usage

### Start FastAPI server

```bash
uvicorn main:app --reload
```

The API will be available at:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Chunking Strategies

| Strategy         | Description |
|------------------|-------------|
| Fixed            | Splits into equal-sized chunks |
| Paragraph        | Keeps text aligned with paragraph boundaries |
| Sentence         | Splits by sentences |
| Sliding Window   | Overlapping chunks for better context |
| Hybrid           | Combines structure + overlap |

---

## ✅ Roadmap

- [ ] Add embeddings + vector database integration (Pinecone, Weaveite, Milvus).

---

## 📜 License

MIT License © 2025
