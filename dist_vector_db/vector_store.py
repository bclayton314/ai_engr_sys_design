import math
from typing import Any


class VectorStore:
    def __init__(self, dimension: int) -> None:
        """
        Initialize an empty in-memory vector store.

        Args:
            dimension: Required dimensionality for all vectors in the store.
        """
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = dimension
        self.records: dict[str, dict[str, Any]] = {}

    def _validate_record(
        self,
        record_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Validate a record before inserting it into the store.
        """
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")

        self._validate_vector(vector)

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def _validate_vector(self, vector: list[float]) -> None:
        """
        Validate a vector against the store's dimensionality requirements.
        """
        if not isinstance(vector, list):
            raise ValueError("vector must be a list")

        if len(vector) != self.dimension:
            raise ValueError(
                f"vector must have dimension {self.dimension}, got {len(vector)}"
            )

        for value in vector:
            if not isinstance(value, (int, float)):
                raise ValueError("vector values must be numeric")

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Convert vector values to floats.
        """
        return [float(v) for v in vector]

    def _cosine_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        cosine_similarity = dot(a, b) / (||a|| * ||b||)
        """
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def upsert(
        self,
        record_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Insert a new record or overwrite an existing one.
        """
        self._validate_record(record_id, vector, text, metadata)

        self.records[record_id] = {
            "id": record_id,
            "vector": self._normalize_vector(vector),
            "text": text,
            "metadata": dict(metadata),
        }

    def get(self, record_id: str) -> dict[str, Any] | None:
        """
        Fetch a record by ID.
        Returns None if not found.
        """
        record = self.records.get(record_id)
        if record is None:
            return None

        return {
            "id": record["id"],
            "vector": list(record["vector"]),
            "text": record["text"],
            "metadata": dict(record["metadata"]),
        }

    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        Returns True if the record existed and was deleted, else False.
        """
        if record_id in self.records:
            del self.records[record_id]
            return True
        return False

    def show_all(self) -> dict[str, dict[str, Any]]:
        """
        Return a copy of all stored records.
        """
        return {
            record_id: {
                "id": record["id"],
                "vector": list(record["vector"]),
                "text": record["text"],
                "metadata": dict(record["metadata"]),
            }
            for record_id, record in self.records.items()
        }

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform exact nearest-neighbor search using cosine similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of top matches to return.

        Returns:
            A list of top-k results sorted by descending similarity score.
        """
        self._validate_vector(query_vector)

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        normalized_query = self._normalize_vector(query_vector)
        scored_results = []

        for record in self.records.values():
            score = self._cosine_similarity(normalized_query, record["vector"])

            scored_results.append({
                "id": record["id"],
                "score": score,
                "text": record["text"],
                "metadata": dict(record["metadata"]),
            })

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:top_k]


def main() -> None:
    store = VectorStore(dimension=4)

    print("Upserting records...")

    store.upsert(
        record_id="doc_001_chunk_0",
        vector=[0.12, -0.44, 0.98, 0.31],
        text="Distributed systems are collections of independent computers.",
        metadata={
            "source": "ddia_notes",
            "topic": "distributed-systems",
            "chunk_index": 0,
        },
    )

    store.upsert(
        record_id="doc_002_chunk_0",
        vector=[0.55, 0.10, -0.25, 0.77],
        text="Vector databases are optimized for similarity search.",
        metadata={
            "source": "ai_engineering_notes",
            "topic": "vector-db",
            "chunk_index": 0,
        },
    )

    store.upsert(
        record_id="doc_003_chunk_0",
        vector=[0.10, -0.40, 0.95, 0.35],
        text="Replication improves durability and availability in distributed systems.",
        metadata={
            "source": "ddia_notes",
            "topic": "replication",
            "chunk_index": 0,
        },
    )

    print("\nFetching one record...")
    print(store.get("doc_001_chunk_0"))

    print("\nRunning vector search...")
    query_vector = [0.11, -0.41, 0.96, 0.30]
    results = store.search(query_vector, top_k=2)

    print("Query vector:", query_vector)
    print("Top results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()