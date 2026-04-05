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

        if not isinstance(vector, list):
            raise ValueError("vector must be a list")

        if len(vector) != self.dimension:
            raise ValueError(
                f"vector must have dimension {self.dimension}, got {len(vector)}"
            )

        for value in vector:
            if not isinstance(value, (int, float)):
                raise ValueError("vector values must be numeric")

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

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
            "vector": [float(v) for v in vector],
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

    print("\nFetching one record...")
    print(store.get("doc_001_chunk_0"))

    print("\nCurrent store contents...")
    print(store.show_all())

    print("\nDeleting one record...")
    deleted = store.delete("doc_002_chunk_0")
    print("Deleted:", deleted)

    print("\nFetching deleted record...")
    print(store.get("doc_002_chunk_0"))

    print("\nFinal store contents...")
    print(store.show_all())


if __name__ == "__main__":
    main()