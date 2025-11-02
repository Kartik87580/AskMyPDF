
# 📄 AskMyPDF

A **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, and **Google Gemini** that lets you **upload a PDF and ask questions** about its contents.

The app automatically:

1. Extracts text from the PDF 🧾
2. Splits it into chunks 🔍
3. Creates or loads a **FAISS vector index** for retrieval 💾
4. Uses **Google Gemini** (via `ChatGoogleGenerativeAI`) to answer user queries based on the PDF context 🤖

---

## 🚀 Features

* 🧠 **LLM-powered QA:** Uses Gemini (`gemini-2.0-flash`) for accurate answers.
* ⚡ **Efficient retrieval:** Caches embeddings and FAISS indexes for faster reuse.
* 📚 **Text chunking:** Automatically splits large PDFs into retrievable pieces.
* 💾 **Local caching:** Reuses vector stores per file to save time and resources.
* 💬 **Streamlit UI:** Clean, interactive interface for uploading PDFs and asking questions.

---

## 🧩 Tech Stack

| Component       | Technology                                                            |
| --------------- | --------------------------------------------------------------------- |
| Frontend/UI     | Streamlit                                                             |
| LLM             | Google Gemini (via `langchain-google-genai`)                          |
| Embeddings      | Sentence Transformers (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`) |
| Vector Store    | FAISS                                                                 |
| Text Processing | LangChain TextSplitter + PyPDF2                                       |

---

## 📁 Project Structure

```
📂 pdf-rag-gemini
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── faiss_index/          # Cached FAISS indexes (auto-created)
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kartik87580/AskMyPDF
cd AskMyPDF
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Google API Key

You need a **Google Gemini API key** for `langchain-google-genai`.

#### Option 1: Using Environment Variable

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

#### Option 2: Using Streamlit Secrets

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the provided URL in your browser (usually `http://localhost:8501`).

---

## 🧠 How It Works

1. **Upload a PDF**

   * The app reads and extracts text using `PyPDF2`.

2. **Chunking**

   * The extracted text is divided into overlapping chunks for semantic search.

3. **Embedding**

   * Each chunk is embedded using a SentenceTransformer model from Hugging Face.

4. **Indexing**

   * FAISS builds a vector store to quickly find the most relevant chunks.

5. **Retrieval + Generation**

   * Relevant chunks are fed into Gemini (`gemini-2.0-flash`) via LangChain’s `RetrievalQA` chain to answer the user’s query.

---

## 🧮 Settings (Sidebar)

| Setting                  | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| **Index Directory**      | Folder to store FAISS indexes                               |
| **Embedding Model**      | Choose from `all-MiniLM-L6-v2` or `paraphrase-MiniLM-L6-v2` |
| **Device**               | Use `cpu` or `cuda`                                         |
| **Chunk Size / Overlap** | Control how text is split for retrieval                     |
| **Retriever k**          | Number of top chunks used for answering                     |

---

## 📦 Requirements

Listed in `requirements.txt`:

```
PyPDF2==3.0.1
langchain==0.3.27
langchain-community
langchain-google-genai
sentence-transformers==5.1.1
faiss-cpu==1.12.0
```

---

## 🧾 Example Usage

1. Upload a PDF (e.g., “research_paper.pdf”)
2. Wait for the app to extract and embed content
3. Ask questions like:

   * *“What is the main conclusion of this paper?”*
   * *“List the key findings in section 3.”*
4. See the AI-generated answer and the relevant source chunks.

---

## 🧰 Troubleshooting

| Issue                          | Solution                                                         |
| ------------------------------ | ---------------------------------------------------------------- |
| **“Google API key not found”** | Add your key to environment or `secrets.toml`                    |
| **“No extractable text”**      | Make sure your PDF contains selectable text (not scanned images) |
| **“FAISS loading failed”**     | Delete the index folder and re-run the app                       |

---

## 🧑‍💻 Author

**Kartik Jambucha**


---
