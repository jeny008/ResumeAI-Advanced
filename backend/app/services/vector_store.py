
from functools import lru_cache
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from app.core.config import (
    QDRANT_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
)


COLLECTION = "resume_chunks"
METADATA_KEY = "metadata"
RESUME_ID_FIELD = "metadata.resume_id"


# -----------------------------------
# 1. Embedding Model
# -----------------------------------

@lru_cache(maxsize=1)
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------------
# 2. Qdrant Client
# -----------------------------------

@lru_cache(maxsize=1)
def client():

    # Qdrant Cloud
    if QDRANT_URL and QDRANT_API_KEY:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )

    # Local Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    return QdrantClient(
        path=str(QDRANT_PATH)
    )


# -----------------------------------
# 3. Create Collection and Index
# -----------------------------------

def ensure_collection():

    qdrant = client()

    # Create collection if it doesn't exist
    if not qdrant.collection_exists(COLLECTION):

        vector_size = len(
            embeddings().embed_query("test")
        )

        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # Create payload index for resume_id filtering
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name=RESUME_ID_FIELD,
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    # Verify index exists
    collection_info = qdrant.get_collection(COLLECTION)

    if RESUME_ID_FIELD not in collection_info.payload_schema:
        raise RuntimeError(
            f"Qdrant index creation failed for {RESUME_ID_FIELD}"
        )


# -----------------------------------
# 4. Get Vector Store
# -----------------------------------

def get_store():

    ensure_collection()

    return QdrantVectorStore(
        client=client(),
        collection_name=COLLECTION,
        embedding=embeddings(),
        content_payload_key="page_content",
        metadata_payload_key=METADATA_KEY,
    )


# -----------------------------------
# 5. Index Resume
# -----------------------------------

def index_resume(resume_id: str, text: str) -> int:

    if not text or not text.strip():
        raise ValueError("Resume text is empty.")

    docs = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    ).create_documents(
        [text],
        metadatas=[
            {"resume_id": resume_id}
        ],
    )

    if not docs:
        raise ValueError(
            "No readable text found in resume."
        )

    store = get_store()

    # Delete existing chunks for this resume
    store.client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key=RESUME_ID_FIELD,
                    match=MatchValue(value=resume_id),
                )
            ]
        ),
        wait=True,
    )

    # Insert new chunks
    ids = [
        str(uuid4())
        for _ in docs
    ]

    store.add_documents(
        documents=docs,
        ids=ids,
    )

    return len(docs)


# -----------------------------------
# 6. Retrieve Resume Chunks
# -----------------------------------

def retrieve(
    resume_id: str,
    query: str,
    k: int = 6,
):

    store = get_store()

    resume_filter = Filter(
        must=[
            FieldCondition(
                key=RESUME_ID_FIELD,
                match=MatchValue(value=resume_id),
            )
        ]
    )

    docs = store.similarity_search(
        query=query,
        k=k,
        filter=resume_filter,
    )

    return [
        doc.page_content
        for doc in docs
    ]
