import hashlib
import os
import tempfile
import warnings
from pathlib import Path

import streamlit as st
import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GenAI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 GenAI RAG Chatbot")
st.caption("Chat with the built-in GenAI guide, or upload your own PDF in the sidebar.")

# ============================================================
# SIDEBAR: MODEL + GENERATION SETTINGS
# ============================================================
MODEL_OPTIONS = {
    "Qwen2.5-0.5B-Instruct (fastest, CPU friendly)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct (original model, needs ~6GB RAM)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (better quality, needs GPU/8GB+ RAM)": "Qwen/Qwen2.5-3B-Instruct",
    "Phi-3.5-mini-instruct (3.8B, strong reasoning)": "microsoft/Phi-3.5-mini-instruct",
}

DEFAULT_PDF = Path(__file__).parent / "GenAI_Fundamentals.pdf"

with st.sidebar:
    st.header("⚙️ Settings")
    model_label = st.selectbox("LLM model", list(MODEL_OPTIONS.keys()), index=0)
    model_id = MODEL_OPTIONS[model_label]

    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05,
                            help="0 = deterministic. Higher = more creative.")
    max_new_tokens = st.slider("Max new tokens", 50, 600, 250, 10)
    top_k_docs = st.slider("Chunks retrieved (k)", 1, 8, 4)
    chunk_size = st.slider("Chunk size", 300, 1500, 750, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)

    uploaded = st.file_uploader("Upload your own PDF (optional)", type=["pdf"])
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading LLM (first time takes a while)...", max_entries=1)
def load_llm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Reading and indexing PDF...")
def build_vector_db(file_bytes: bytes, file_hash: str, chunk_size: int, chunk_overlap: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=splits,
        embedding=load_embeddings(),
        collection_name=f"pdf_{file_hash[:16]}_{chunk_size}_{chunk_overlap}",
    )  # in-memory (no persist_directory) so it works on any host
    return db, len(docs), len(splits)


# ============================================================
# RAG HELPERS
# ============================================================
def format_docs(docs):
    return "\n\n".join(" ".join(d.page_content.split()) for d in docs)


def generate_answer(tokenizer, model, question, context, temperature, max_new_tokens):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question strictly using the "
                "provided context. If the context does not contain the answer, say "
                "you don't know. Keep your answer clear and concise."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.1,
                      pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:  # temperature only works when do_sample=True
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ============================================================
# MAIN APP
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded is not None:
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
elif DEFAULT_PDF.exists():
    file_bytes, file_name = DEFAULT_PDF.read_bytes(), DEFAULT_PDF.name
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

file_hash = hashlib.md5(file_bytes).hexdigest()
vector_db, n_pages, n_chunks = build_vector_db(file_bytes, file_hash, chunk_size, chunk_overlap)
st.sidebar.success(f"📄 {file_name}\n\n{n_pages} pages → {n_chunks} chunks")

tokenizer, model = load_llm(model_id)

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(s)

# New question
if question := st.chat_input("Ask something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = vector_db.as_retriever(search_kwargs={"k": top_k_docs}).invoke(question)
            answer = generate_answer(tokenizer, model, question, format_docs(docs),
                                     temperature, max_new_tokens)
        st.markdown(answer)
        sources = [
            f"Page {d.metadata.get('page', 0) + 1}: " + " ".join(d.page_content.split())[:250] + "..."
            for d in docs
        ]
        with st.expander("Sources"):
            for s in sources:
                st.caption(s)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
