import os
import secrets
import threading

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Flask binds immediately; RAG stack loads on first question (can take minutes on CPU).
_qa = None
_qa_lock = threading.Lock()

FLASK_PORT = int(os.getenv("FLASK_PORT", "5050"))
# How many vector chunks to retrieve (similarity search only — not "all PDFs"). Higher = more context + more citations; slower / more tokens.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))


def _format_citations_from_docs(docs):
    """Build readable source lines from LangChain Document.metadata (e.g. PyPDFLoader: source, page)."""
    seen = set()
    lines = []
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        src = meta.get("source") or ""
        base = os.path.basename(src) if src else "Unknown source"
        page = meta.get("page")
        key = (base, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            try:
                p = int(page) + 1
                lines.append(f"{base} (page {p})")
            except (TypeError, ValueError):
                lines.append(base)
        else:
            lines.append(base)
    return lines


def get_qa():
    global _qa
    with _qa_lock:
        if _qa is not None:
            return _qa
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_community.llms import HuggingFacePipeline
        from langchain_community.vectorstores import Pinecone
        from pinecone import Pinecone as PineconeSdk

        from src.helper import (
            build_hf_text_generation_pipeline,
            download_hugging_face_embeddings,
        )
        from src.prompt import prompt_template

        index_name = "chatbot"
        embeddings = download_hugging_face_embeddings()

        PINECONE_TIMEOUT_SEC = int(os.getenv("PINECONE_TIMEOUT_SEC", "300"))
        _pc = PineconeSdk(
            api_key=PINECONE_API_KEY,
            timeout=PINECONE_TIMEOUT_SEC,
            pool_threads=8,
        )
        _pinecone_index = _pc.Index(index_name)
        vectorstore = Pinecone(_pinecone_index, embeddings, text_key="text")

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        chain_type_kwargs = {"prompt": PROMPT}

        pipe = build_hf_text_generation_pipeline()
        llm = HuggingFacePipeline(pipeline=pipe)
        _qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        return _qa


@app.route("/health")
def health():
    """Hit this first to confirm the server is listening (no ML load)."""
    return "ok", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/")
def index():
    """Main entry: use the chat UI template."""
    return redirect(url_for("chat"))


_AV_BOT = "https://randomuser.me/api/portraits/women/44.jpg"
_AV_USER = "https://randomuser.me/api/portraits/men/32.jpg"


def _chat_rows_for_template():
    rows = list(session.get("conversation") or [])
    if not rows:
        return [
            {
                "avatar": _AV_BOT,
                "name": "FRP bot",
                "messages": [
                    "Ask about FRP / resins / bars. Each reply lists **Sources** — the PDF pages used as context (similarity search, not the whole library at once). First reply may take several minutes on CPU while the model loads."
                ],
                "type": "them",
            }
        ]
    return rows


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        if request.form.get("action") == "clear":
            session.pop("conversation", None)
            session.modified = True
            return redirect(url_for("chat"))
        msg = (request.form.get("message") or "").strip()
        if msg:
            conv = list(session.get("conversation") or [])
            conv.append(
                {
                    "avatar": _AV_USER,
                    "name": "You",
                    "messages": [msg],
                    "type": "me",
                }
            )
            out = get_qa().invoke({"query": msg})
            reply = out.get("result") or ""
            citations = _format_citations_from_docs(out.get("source_documents"))
            bot_row = {
                "avatar": _AV_BOT,
                "name": "FRP bot",
                "messages": [reply],
                "type": "them",
            }
            if citations:
                bot_row["citations"] = citations
            conv.append(bot_row)
            session["conversation"] = conv[-20:]
            session.modified = True
            return redirect(url_for("chat"))
    return render_template("chat.html", conversation=_chat_rows_for_template())


if __name__ == "__main__":
    print(
        f"\n>>> Server starting on port {FLASK_PORT} (models load on first question).\n"
        f"    Test:  http://127.0.0.1:{FLASK_PORT}/health   → should show: ok\n"
        f"    Chat UI: http://127.0.0.1:{FLASK_PORT}/  (redirects to /chat)\n"
        "    On Windows, use 127.0.0.1 — not 'localhost' — if the page does not load.\n"
    )
    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=True,
        use_reloader=False,
        threaded=True,
    )
