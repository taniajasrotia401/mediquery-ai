\# 🏥 MediQuery AI — Medical Document Intelligence Assistant



A RAG-powered AI assistant that answers medical questions from real clinical transcriptions with source citations.



\## 🎯 Live Demo

\[Coming soon — Streamlit Cloud]



\## 🧠 How It Works

1\. 5000+ real medical transcriptions loaded from MTSamples dataset

2\. Documents split into chunks and embedded using HuggingFace

3\. FAISS vector store enables lightning-fast similarity search

4\. Llama 3 (70B) generates answers grounded in retrieved medical records

5\. Every answer includes source citations with specialty and report name



\## 🛠️ Tech Stack

| Component | Technology |

|---|---|

| LLM | Llama 3.3 70B via Groq API (free) |

| Framework | LangChain |

| Embeddings | HuggingFace all-MiniLM-L6-v2 |

| Vector Store | FAISS |

| UI | Streamlit |

| Dataset | MTSamples (5000 medical transcriptions) |



\## 🚀 Run Locally

```bash

git clone https://github.com/taniajasrotia401/mediquery-ai.git

cd mediquery-ai

pip install -r requirements.txt

streamlit run app.py

```



\## 💡 Example Questions

\- "Describe a cardiovascular surgery procedure"

\- "What medications are used for neurology patients?"

\- "What does a hematology consult involve?"



\## 📁 Project Structure

```

mediquery-ai/

├── app.py              # Streamlit UI + RAG pipeline

├── requirements.txt    # Dependencies

└── README.md           # You are here

```



\## 🔑 Environment Variables

Create a `.env` file:

```

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



\## 👩‍💻 Author

Tania — Data Science fresher passionate about AI in healthcare

