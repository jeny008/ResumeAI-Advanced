
from functools import lru_cache
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.core.config import QDRANT_PATH


COLLECTION = "resume_chunks"


@lru_cache(maxsize=1)
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@lru_cache(maxsize=1)
def client():
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection():
    qdrant = client()

    # Create collection automatically if it does not exist
    if not qdrant.collection_exists(collection_name=COLLECTION):
        vector_size = len(embeddings().embed_query("test"))

        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def get_store():
    ensure_collection()

    return Qdrant(
        client=client(),
        collection_name=COLLECTION,
        embeddings=embeddings()
    )


def index_resume(resume_id: str, text: str) -> int:

    docs = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    ).create_documents(
        [text],
        metadatas=[{"resume_id": resume_id}]
    )

    if not docs:
        raise ValueError("No text found in resume. Please upload a readable PDF, DOCX or TXT file.")

    store = get_store()

    # Delete old chunks for this resume, if any
    store.client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.resume_id",
                    match=MatchValue(value=resume_id)
                )
            ]
        )
    )

    # Insert new resume chunks
    store.add_documents(
        docs,
        ids=[str(uuid4()) for _ in docs]
    )

    return len(docs)



def retrieve(resume_id: str, query: str, k: int = 6):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    store = get_store()

    # Build a proper Qdrant metadata filter
    resume_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.resume_id",
                match=MatchValue(value=resume_id)
            )
        ]
    )

    # Retrieve only chunks belonging to this resume
    docs = store.similarity_search(
        query=query,
        k=k,
        filter=resume_filter
    )

    return [doc.page_content for doc in docs]
