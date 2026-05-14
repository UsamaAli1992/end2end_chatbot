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


def build_hf_text_generation_pipeline():
    """Local text-generation pipeline (same strategy as research/trials.ipynb)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    local_llama = Path("model") / "Llama-3.2-3B-Instruct"
    cpu_light = "Qwen/Qwen2.5-0.5B-Instruct"
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_id = str(local_llama)
        dtype = torch.float16
        load_kw = {"torch_dtype": dtype, "device_map": "auto"}
    else:
        model_id = cpu_light
        dtype = torch.float32
        load_kw = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

    print(
        "LLM:",
        "CUDA → local Llama 3.2 3B" if use_cuda else "CPU → lightweight " + cpu_light,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, **load_kw
    )
    if not use_cuda:
        hf_model = hf_model.to("cpu")

    return pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.15,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )