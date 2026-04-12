import json
import math
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAL_PATH = Path(SCRIPT_DIR) / "vector_store.wal"
SNAPSHOT_PATH = Path(SCRIPT_DIR) / "vector_store.snapshot.json"

HOST = "127.0.0.1"
PORT = 8080
VECTOR_DIMENSION = 8


class MockEmbedder:
    """
    A simple deterministic text embedder for learning purposes.

    This is NOT a real semantic embedding model.
    It turns text into a fixed-size numeric vector by hashing tokens into buckets.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        tokens = text.lower().split()
        vector = [0.0] * self.dimension

        if not tokens:
            return vector

        token_counts = Counter(tokens)

        for token, count in token_counts.items():
            bucket = sum(ord(ch) for ch in token) % self.dimension
            vector[bucket] += float(count)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector

        return [v / norm for v in vector]


class VectorStore:
    def __init__(self, dimension: int, wal_path: Path, snapshot_path: Path) -> None:
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
        snapshot_data = {
            "records": self.show_all(),
            "last_wal_line": self._count_wal_lines(),
        }

        with self.snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2)

    def load_snapshot_contents(self) -> dict[str, Any] | None:
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


embedder = MockEmbedder(dimension=VECTOR_DIMENSION)
vector_store = VectorStore(
    dimension=VECTOR_DIMENSION,
    wal_path=WAL_PATH,
    snapshot_path=SNAPSHOT_PATH,
)


class VectorStoreRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        response_body = json.dumps(payload).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")

        if not body:
            return {}

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON body: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")

        return data

    def _extract_record_id_from_path(self) -> str | None:
        parsed = urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) == 2 and path_parts[0] == "vectors":
            return path_parts[1]

        return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/vectors":
            self._send_json(200, {"records": vector_store.show_all()})
            return

        if parsed.path == "/snapshot":
            snapshot = vector_store.load_snapshot_contents()
            if snapshot is None:
                self._send_json(404, {"error": "No snapshot found"})
            else:
                self._send_json(200, {"snapshot": snapshot})
            return

        record_id = self._extract_record_id_from_path()
        if record_id is not None:
            record = vector_store.get(record_id)
            if record is None:
                self._send_json(404, {"error": f"Record '{record_id}' not found"})
            else:
                self._send_json(200, {"record": record})
            return

        self._send_json(404, {"error": "Route not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/vectors/upsert":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            record_id = body.get("id")
            vector = body.get("vector")
            text = body.get("text", "")
            metadata = body.get("metadata", {})

            if record_id is None or vector is None:
                self._send_json(400, {"error": "Missing required fields: 'id' and 'vector'"})
                return

            try:
                vector_store.upsert(
                    record_id=record_id,
                    vector=vector,
                    text=text,
                    metadata=metadata,
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            self._send_json(200, {
                "message": "record upserted",
                "id": record_id,
                "mode": "vector_upsert",
            })
            return

        if parsed.path == "/documents/upsert":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            record_id = body.get("id")
            text = body.get("text")
            metadata = body.get("metadata", {})

            if record_id is None or text is None:
                self._send_json(400, {"error": "Missing required fields: 'id' and 'text'"})
                return

            if not isinstance(text, str):
                self._send_json(400, {"error": "'text' must be a string"})
                return

            try:
                vector = embedder.embed(text)
                vector_store.upsert(
                    record_id=record_id,
                    vector=vector,
                    text=text,
                    metadata=metadata,
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            self._send_json(200, {
                "message": "document embedded and upserted",
                "id": record_id,
                "vector_dimension": len(vector),
                "mode": "document_upsert",
            })
            return

        if parsed.path == "/search":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            query_vector = body.get("query_vector")
            top_k = body.get("top_k", 5)
            filters = body.get("filters")

            if query_vector is None:
                self._send_json(400, {"error": "Missing required field: 'query_vector'"})
                return

            try:
                results = vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    filters=filters,
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            self._send_json(200, {
                "results": results,
                "top_k": top_k,
                "filters": filters,
                "mode": "vector_search",
            })
            return

        if parsed.path == "/documents/search":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            query_text = body.get("query_text")
            top_k = body.get("top_k", 5)
            filters = body.get("filters")

            if query_text is None:
                self._send_json(400, {"error": "Missing required field: 'query_text'"})
                return

            if not isinstance(query_text, str):
                self._send_json(400, {"error": "'query_text' must be a string"})
                return

            try:
                query_vector = embedder.embed(query_text)
                results = vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    filters=filters,
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            self._send_json(200, {
                "query_text": query_text,
                "results": results,
                "top_k": top_k,
                "filters": filters,
                "mode": "document_search",
            })
            return

        if parsed.path == "/snapshot":
            vector_store.create_snapshot()
            self._send_json(200, {"message": "snapshot created"})
            return

        self._send_json(404, {"error": "Route not found"})

    def do_DELETE(self) -> None:
        record_id = self._extract_record_id_from_path()
        if record_id is None:
            self._send_json(404, {"error": "Route not found"})
            return

        deleted = vector_store.delete(record_id)
        if deleted:
            self._send_json(200, {
                "message": "record deleted",
                "id": record_id,
            })
        else:
            self._send_json(404, {"error": f"Record '{record_id}' not found"})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.command} {self.path} - {format % args}")


def run_server() -> None:
    server = HTTPServer((HOST, PORT), VectorStoreRequestHandler)
    print(f"Vector store HTTP server running at http://{HOST}:{PORT}")
    print(f"Embedding mode: mock embedder (dimension={VECTOR_DIMENSION})")
    print("Routes:")
    print("  GET    /vectors")
    print("  GET    /vectors/<id>")
    print("  DELETE /vectors/<id>")
    print("  POST   /vectors/upsert")
    print("  POST   /documents/upsert")
    print("  POST   /search")
    print("  POST   /documents/search")
    print("  POST   /snapshot")
    print("  GET    /snapshot")
    server.serve_forever()


if __name__ == "__main__":
    run_server()