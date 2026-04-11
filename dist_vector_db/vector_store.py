import json
import math
from pathlib import Path
from typing import Any


WAL_PATH = Path("vector_store.wal")


class VectorStore:
    def __init__(self, dimension: int, wal_path: Path) -> None:
        """
        Initialize an in-memory vector store with WAL-based recovery.

        Args:
            dimension: Required dimensionality for all vectors in the store.
            wal_path: Path to the write-ahead log file.
        """
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = dimension
        self.records: dict[str, dict[str, Any]] = {}
        self.wal_path = wal_path

        self.wal_path.touch(exist_ok=True)
        self.replay_wal()

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
        """
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _metadata_matches(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        """
        Return True if a record's metadata satisfies all requested filters.
        """
        if filters is None:
            return True

        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary or None")

        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != expected_value:
                return False

        return True

    def _append_to_wal(self, record: dict[str, Any]) -> None:
        """
        Append one mutation record to the write-ahead log.
        """
        with self.wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def replay_wal(self) -> None:
        """
        Rebuild in-memory state by replaying WAL operations in order.
        """
        with self.wal_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in WAL at line {line_number}: {e}"
                    ) from e

                op = record.get("op")
                record_id = record.get("id")

                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(
                        f"Invalid or missing record id in WAL at line {line_number}: {record}"
                    )

                if op == "UPSERT":
                    vector = record.get("vector")
                    text = record.get("text")
                    metadata = record.get("metadata")

                    if vector is None or text is None or metadata is None:
                        raise ValueError(
                            f"Invalid UPSERT record in WAL at line {line_number}: {record}"
                        )

                    self._validate_record(record_id, vector, text, metadata)

                    self.records[record_id] = {
                        "id": record_id,
                        "vector": self._normalize_vector(vector),
                        "text": text,
                        "metadata": dict(metadata),
                    }

                elif op == "DELETE":
                    self.records.pop(record_id, None)

                else:
                    raise ValueError(
                        f"Unknown WAL operation at line {line_number}: {record}"
                    )

    def upsert(
        self,
        record_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Insert a new record or overwrite an existing one.
        Logs the mutation before applying it to memory.
        """
        self._validate_record(record_id, vector, text, metadata)

        normalized_vector = self._normalize_vector(vector)

        wal_record = {
            "op": "UPSERT",
            "id": record_id,
            "vector": normalized_vector,
            "text": text,
            "metadata": dict(metadata),
        }
        self._append_to_wal(wal_record)

        self.records[record_id] = {
            "id": record_id,
            "vector": normalized_vector,
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
        Logs the deletion before applying it to memory.
        Returns True if the record existed and was deleted, else False.
        """
        if record_id not in self.records:
            return False

        wal_record = {
            "op": "DELETE",
            "id": record_id,
        }
        self._append_to_wal(wal_record)

        del self.records[record_id]
        return True

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

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform exact nearest-neighbor search using cosine similarity,
        optionally applying metadata filters before scoring.
        """
        self._validate_vector(query_vector)

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        normalized_query = self._normalize_vector(query_vector)
        scored_results = []

        for record in self.records.values():
            if not self._metadata_matches(record["metadata"], filters):
                continue

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
    store = VectorStore(dimension=4, wal_path=WAL_PATH)

    print("Current recovered records at startup:")
    print(store.show_all())

    print("\nUpserting records...")

    store.upsert(
        record_id="doc_001_chunk_0",
        vector=[0.12, -0.44, 0.98, 0.31],
        text="Distributed systems are collections of independent computers.",
        metadata={
            "source": "ddia_notes",
            "topic": "distributed-systems",
            "chunk_index": 0,
            "language": "en",
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
            "language": "en",
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
            "language": "en",
        },
    )

    print("\nSearch without filters:")
    query_vector = [0.11, -0.41, 0.96, 0.30]
    results = store.search(query_vector, top_k=2)
    for result in results:
        print(result)

    print("\nDeleting one record...")
    deleted = store.delete("doc_002_chunk_0")
    print("Deleted:", deleted)

    print("\nCurrent records...")
    print(store.show_all())

    print(f"\nWAL file written to: {WAL_PATH}")
    print("Restart the program to observe WAL-based recovery.")


if __name__ == "__main__":
    main()