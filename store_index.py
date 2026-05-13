from src.helper import (
    load_all_pdf_documents,
    text_split,
    download_hugging_face_embeddings,
)
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY is not set")

# Pinecone v6+: no pinecone.init(). LangChain reads PINECONE_API_KEY from the environment.
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

index_name = "chatbot"

# False = load all PDFs, chunk, embed, upsert (slow). True = only attach to existing Pinecone index.
# Override without editing: set env SKIP_EXPENSIVE_INGEST=false
_env_skip = os.getenv("SKIP_EXPENSIVE_INGEST", "").strip().lower()
if _env_skip in ("0", "false", "no"):
    SKIP_EXPENSIVE_INGEST = False
elif _env_skip in ("1", "true", "yes"):
    SKIP_EXPENSIVE_INGEST = True
else:
    SKIP_EXPENSIVE_INGEST = True  # default: skip long ingest

if SKIP_EXPENSIVE_INGEST:
    extracted_data = []
else:
    extracted_data = load_all_pdf_documents("data")

    # One LangChain Document per PDF page (typical); many pages ⇒ many rows.
    print(f"Total loaded chunks: {len(extracted_data)}")
    pdf_files = sorted({os.path.basename(d.metadata["source"]) for d in extracted_data})
    print(f"Unique PDF files represented: {len(pdf_files)}")
    print(pdf_files[:20], "..." if len(pdf_files) > 20 else "")

if SKIP_EXPENSIVE_INGEST:
    text_chunks = []
else:
    text_chunks = text_split(extracted_data)

    print(f"Documents from PDFs (pages): {len(extracted_data)}")
    print(f"Chunks after RecursiveCharacterTextSplitter: {len(text_chunks)}")


embeddings = download_hugging_face_embeddings()

# Create / refresh vector index in Pinecone
# Embed each chunk and upsert into Pinecone (tutorial-style).
# Full ~33k chunks takes a long time — set TEST_LIMIT to a small number first.
# LangChain's from_texts() uses a 30s HTTP timeout; slow/VPN networks can hit SSL handshake timeouts.
# Here we build the client with a longer timeout, then add_texts (same end result as from_texts).

TEST_LIMIT = None  # e.g. 200 for a dry run; None = upload all chunks_to_upload

from pinecone import Pinecone as PineconeSdk

PINECONE_TIMEOUT_SEC = 180  # increase (e.g. 300) if PineconeTimeoutError persists

if SKIP_EXPENSIVE_INGEST:
    # Index already populated — reconnect only (no duplicate upsert).
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
    )
else:
    chunks_to_upload = text_chunks if TEST_LIMIT is None else text_chunks[:TEST_LIMIT]

    _pc = PineconeSdk(
        api_key=os.environ["PINECONE_API_KEY"],
        timeout=PINECONE_TIMEOUT_SEC,
        pool_threads=8,
    )
    _pinecone_index = _pc.Index(index_name)

    vectorstore = Pinecone(_pinecone_index, embeddings, text_key="text")
    vectorstore.add_texts(
        [t.page_content for t in chunks_to_upload],
        metadatas=[dict(t.metadata) for t in chunks_to_upload],
    )

print("Done. vectorstore:", vectorstore)
