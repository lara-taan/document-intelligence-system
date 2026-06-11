import streamlit as st
import tempfile
from rag_pipeline import parse_pdf, build_vector_store, get_answer

st.set_page_config(page_title="Document Q&A")
st.title("Document Q&A — Ask Your PDF")
st.write("Upload a PDF and ask questions about its content.")

# Upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:

    # Save to a temp file so pdfplumber can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Process only once per upload
    if "vector_store" not in st.session_state:
        with st.spinner("Reading and indexing your document..."):
            text = parse_pdf(tmp_path)
            st.session_state.vector_store = build_vector_store(text)
        st.success("Document ready! Ask your questions below.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show sources if they exist
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 View sources from document"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(source)
                        st.divider()

    # Chat input at the bottom
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Save user message to history
        st.session_state.messages.append({"role": "user", "content": question})

        # Get and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_answer(st.session_state.vector_store, question)
            st.write(result["answer"])
            with st.expander("📎 View sources from document"):
                for i, source in enumerate(result["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(source)
                    st.divider()

        # Save assistant message + sources to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

    # Reset button
    if st.button("Upload a different document"):
        del st.session_state.vector_store
        del st.session_state.messages
        st.rerun()
