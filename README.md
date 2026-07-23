# 📄 CourseMate AI

An AI-powered PDF Q&A assistant. Upload any PDF, and ask questions about it in a clean chat interface — answers are generated strictly from the content of your document using Retrieval-Augmented Generation (RAG).

<div align="center">

### [🚀 &nbsp;**Try the Live Demo**&nbsp; 🚀](https://yhzrayayzpvlehbefx5cuu.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yhzrayayzpvlehbefx5cuu.streamlit.app/)

</div>

<!-- Replace the line below with an actual screenshot of your app -->
![App Screenshot](docs/screenshot.png)

---

## ✨ Features

- 📎 **Drag-and-drop PDF upload** — no pre-processing or setup required
- 🔍 **Automatic chunking & embedding** using `sentence-transformers/all-MiniLM-L6-v2`
- 💬 **Chat interface** with full conversation history for the session
- 🎯 **Grounded answers** — the model is instructed to only answer from the document, and says so clearly if the answer isn't present
- 📊 **Document stats** — page count and chunk count shown after processing
- 🎨 **Custom dark UI** with a polished, modern look

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Mistral AI](https://mistral.ai/) (`mistral-small-latest`) via `langchain-mistralai` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [Chroma](https://www.trychroma.com/) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| PDF Parsing | `pypdf` |

## 🧠 How It Works

1. **Upload** a PDF through the sidebar.
2. The document is split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`).
3. Each chunk is embedded and stored in an in-memory Chroma vector store.
4. When you ask a question, the app retrieves the most relevant chunks (MMR search) and passes them as context to Mistral's chat model.
5. The model answers **using only that retrieved context** — reducing hallucination.

## 🚀 Running Locally

```bash
git clone https://github.com/rajputshivamsingh510/coursemate-ai.git
cd coursemate-ai
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your-mistral-api-key-here
```

Then run:
```bash
streamlit run app.py
```

## ☁️ Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own copy:

1. Fork this repo.
2. Connect it on [share.streamlit.io](https://share.streamlit.io/).
3. Add your `MISTRAL_API_KEY` under **App Settings → Secrets**:
   ```toml
   MISTRAL_API_KEY = "your-key-here"
   ```

## ⚠️ Notes & Limitations

- The vector store is **in-memory only** — uploaded documents and chat history reset when the app restarts or goes idle. There's no persistent storage between sessions.
- Answers are only as good as the retrieved chunks; very long or image-heavy PDFs may not chunk ideally.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
