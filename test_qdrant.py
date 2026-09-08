from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    url="http://localhost:6333",
    timeout=60
)

if not client.collection_exists("nimbus_docs"):
    client.create_collection(
        collection_name="nimbus_docs",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

print(client.get_collection("nimbus_docs"))