from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_all_pdf_documents(data_dir: str | Path, recursive: bool = False):
    """Load every PDF under `data_dir`.

    Renamed from load_pdfs so a past typo (`load_pdfs = "data/"`) cannot shadow this function in the kernel.

    - recursive=False: only *.pdf in that folder (same as your sample).
    - recursive=True: also PDFs in subfolders (**/*.pdf).
    """
    root = Path(data_dir).resolve()
    pattern = "**/*.pdf" if recursive else "*.pdf"

    loader = DirectoryLoader(
        str(root),
        glob=pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,  # progress bar when loading many files
    )
    return loader.load()






def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



# Sentence-transformers model — downloaded/cached from Hugging Face Hub on first use (~90MB).
def download_hugging_face_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Return a LangChain HuggingFaceEmbeddings instance for chunk vectors."""
    return HuggingFaceEmbeddings(model_name=model_name)