import os
import streamlit as st
from dotenv import load_dotenv
from ingest import build_index
from rag_deploy import rag_query

load_dotenv()

# ---------------- Page Config ----------------
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚")

st.title("📚 RAG-based Research Paper Assistant")
st.caption("Upload PDFs, build a vector index, and ask questions grounded in your documents.")

# ---------------- Step 1: Upload PDFs ----------------
with st.expander("Step 1 — Upload PDFs"):
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        os.makedirs("data", exist_ok=True)
        saved_paths = []
        for f in uploaded:
            path = os.path.join("data", f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            saved_paths.append(path)
        st.success(f"✅ Saved {len(saved_paths)} PDFs to ./data")

    if st.button("Build/Refresh Index"):
        pdf_dir = "data"
        pdfs = [os.path.join(pdf_dir, p) for p in os.listdir(pdf_dir) if p.lower().endswith(".pdf")]
        if not pdfs:
            st.error("❌ No PDFs found in ./data. Upload first.")
        else:
            with st.spinner("⚡ Building FAISS index..."):
                build_index(pdfs, out_dir="vector_store")
            st.success("🎉 Index built successfully!")

st.markdown("---")

# ---------------- Step 2: Ask Question ----------------
st.subheader("Step 2 — Ask a question")

# Model selector
generator = st.selectbox(
    "Choose model for answering",
    ["ollama", "gemini"],  # matches rag.py
    index=1                # default = gemini (better for cloud)
)

query = st.text_input("🔎 Your question about the PDFs")
top_k = st.slider("📑 Top K passages", 3, 10, 5)

if st.button("Get Answer") and query:
    with st.spinner("🤔 Retrieving and generating..."):
        try:
            answer, hits = rag_query(
                query,
                store_dir="vector_store",
                top_k=top_k,
                generator=generator
            )

            # Show answer
            st.markdown("### 🧾 Answer")
            st.write(answer)

            # Show context
            with st.expander("Show retrieved context"):
                for h in hits:
                    st.markdown(f"**{h['source']} • chunk {h['chunk']} • score {h['score']:.3f}**")
                    st.write(h["text"])
                    st.markdown("---")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
