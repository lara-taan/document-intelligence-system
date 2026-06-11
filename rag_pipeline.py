import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def parse_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def build_vector_store(text: str):
    """Split text into chunks and embed them using FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    return vector_store


def get_answer(vector_store, question: str) -> dict:
    """Find relevant chunks and generate a grounded answer."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    If the answer is not in the context, say "I couldn't find that in the document."

    Context: {context}

    Question: {question}
    """)

    source_docs = retriever.invoke(question)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in source_docs]
    }
