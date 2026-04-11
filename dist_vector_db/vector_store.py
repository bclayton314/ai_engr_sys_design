import json
import math
from pathlib import Path
from typing import Any
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAL_PATH = Path(SCRIPT_DIR) / "vector_store.wal"
SNAPSHOT_PATH = Path(SCRIPT_DIR) / "vector_store.snapshot.json"


class VectorStore:
    def __init__(self, dimension: int, wal_path: Path, snapshot_path: Path) -> None:
        """
        Initialize an in-memory vector store with snapshot + WAL recovery.

        Args:
            dimension: Required dimensionality for all vectors in the store.
            wal_path: Path to the write-ahead log file.
            snapshot_path: Path to the snapshot file.
        """
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = dimension
        self.records: dict[str, dict[str, Any]] = {}
        self.wal_path = wal_path
        self.snapshot_path = snapshot_path

        self.wal_path.touch(exist_ok=True)
        self.recover()

    def _validate_record(
        self,
        record_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")

        self._validate_vector(vector)

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def _validate_vector(self, vector: list[float]) -> None:
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
        return [float(v) for v in vector]

    def _cosine_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
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
        with self.wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _count_wal_lines(self) -> int:
        count = 0
        with self.wal_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _load_snapshot(self) -> tuple[dict[str, dict[str, Any]], int]:
        """
        Load snapshot records and the last included WAL line.
        Returns ({}, 0) if no snapshot exists.
        """
        if not self.snapshot_path.exists():
            return {}, 0

        raw_text = self.snapshot_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return {}, 0

        try:
            snapshot_data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in snapshot file {self.snapshot_path}: {e}"
            ) from e

        if not isinstance(snapshot_data, dict):
            raise ValueError(
                f"Snapshot file must contain a JSON object: {self.snapshot_path}"
            )

        records = snapshot_data.get("records", {})
        last_wal_line = snapshot_data.get("last_wal_line", 0)

        if not isinstance(records, dict):
            raise ValueError("Snapshot field 'records' must be a JSON object.")

        if not isinstance(last_wal_line, int) or last_wal_line < 0:
            raise ValueError(
                "Snapshot field 'last_wal_line' must be a non-negative integer."
            )

        for record_id, record in records.items():
            if not isinstance(record, dict):
                raise ValueError(f"Snapshot record for '{record_id}' must be an object.")

            vector = record.get("vector")
            text = record.get("text")
            metadata = record.get("metadata")

            self._validate_record(record_id, vector, text, metadata)

        return records, last_wal_line

    def _replay_wal_from_line(self, start_line: int) -> None:
        """
        Replay WAL records starting after start_line.
        """
        with self.wal_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if line_number <= start_line:
                    continue

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

    def recover(self) -> None:
        """
        Recover using:
        1. snapshot
        2. WAL tail after snapshot
        """
        snapshot_records, last_wal_line = self._load_snapshot()

        self.records = {}
        for record_id, record in snapshot_records.items():
            self.records[record_id] = {
                "id": record["id"],
                "vector": self._normalize_vector(record["vector"]),
                "text": record["text"],
                "metadata": dict(record["metadata"]),
            }

        self._replay_wal_from_line(last_wal_line)

    def create_snapshot(self) -> None:
        """
        Save the current full vector state and the latest WAL line count.
        """
        snapshot_data = {
            "records": self.show_all(),
            "last_wal_line": self._count_wal_lines(),
        }

        with self.snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2)

    def load_snapshot_contents(self) -> dict[str, Any] | None:
        """
        Read the raw snapshot file for inspection.
        Returns None if no snapshot exists yet.
        """
        if not self.snapshot_path.exists():
            return None

        raw_text = self.snapshot_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return None

        try:
            snapshot_data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in snapshot file {self.snapshot_path}: {e}"
            ) from e

        if not isinstance(snapshot_data, dict):
            raise ValueError(
                f"Snapshot file must contain a JSON object: {self.snapshot_path}"
            )

        return snapshot_data

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
    store = VectorStore(
        dimension=4,
        wal_path=WAL_PATH,
        snapshot_path=SNAPSHOT_PATH,
    )

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

    print("\nCreating snapshot...")
    store.create_snapshot()
    print(f"Snapshot written to: {SNAPSHOT_PATH}")

    print("\nUpserting one more record after snapshot...")
    store.upsert(
        record_id="doc_004_chunk_0",
        vector=[0.11, -0.39, 0.94, 0.33],
        text="Partitioning and replication are key distributed systems concepts.",
        metadata={
            "source": "ddia_notes",
            "topic": "replication",
            "chunk_index": 1,
            "language": "en",
        },
    )

    print("\nSearch filtered by topic='replication':")
    query_vector = [0.11, -0.41, 0.96, 0.30]
    results = store.search(query_vector, top_k=3, filters={"topic": "replication"})
    for result in results:
        print(result)

    print("\nCurrent records...")
    print(store.show_all())

    print("\nSnapshot contents:")
    print(store.load_snapshot_contents())

    print("\nRestart the program to observe snapshot + WAL-tail recovery.")


if __name__ == "__main__":
    main()