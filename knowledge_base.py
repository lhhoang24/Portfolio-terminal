from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "knowledge_base.sqlite3"
DEFAULT_CHAT_MODEL = "qwen3:1.7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaError(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        chunks INTEGER NOT NULL DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        location TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def ollama_status(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return True, "Đã kết nối Ollama. Mô hình hiện có: " + (", ".join(models) if models else "chưa tải mô hình")
    except requests.RequestException as exc:
        return False, f"Không kết nối được Ollama tại {base_url}: {exc}"


def _embed(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = requests.post(
            base_url.rstrip("/") + "/api/embed",
            json={"model": model, "input": texts, "keep_alive": "30m"}, timeout=180,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError("Ollama không trả về đủ embedding. Kiểm tra tên embedding model.")
        return vectors
    except requests.RequestException as exc:
        raise OllamaError(f"Không gọi được embedding model '{model}': {exc}") from exc


def _extract_text(filename: str, raw: bytes) -> list[tuple[str, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            return [(f"Trang {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        except Exception as exc:
            raise ValueError(f"Không đọc được PDF: {exc}") from exc
    if suffix == ".docx":
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(raw))
            blocks = [p.text for p in doc.paragraphs if p.text.strip()]
            for ti, table in enumerate(doc.tables, 1):
                for row in table.rows:
                    blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
            return [("Nội dung DOCX", "\n".join(blocks))]
        except Exception as exc:
            raise ValueError(f"Không đọc được DOCX: {exc}") from exc
    if suffix == ".pptx":
        try:
            from pptx import Presentation
            import io
            prs = Presentation(io.BytesIO(raw))
            result = []
            for i, slide in enumerate(prs.slides, 1):
                parts = [shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
                result.append((f"Slide {i}", "\n".join(parts)))
            return result
        except Exception as exc:
            raise ValueError(f"Không đọc được PPTX: {exc}") from exc
    if suffix in {".txt", ".md", ".csv"}:
        for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                return [("Nội dung", raw.decode(encoding))]
            except UnicodeDecodeError:
                continue
    raise ValueError("Định dạng không được hỗ trợ. Hãy dùng PDF, DOCX, PPTX, TXT, MD hoặc CSV.")


def _chunk_pages(pages: list[tuple[str, str]], size: int = 1100, overlap: int = 160) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for location, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + int(size * .65), end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if len(piece) >= 35:
                chunks.append((location, piece))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def add_document(filename: str, raw: bytes, base_url: str, embed_model: str) -> tuple[int, str]:
    digest = hashlib.sha256(raw).hexdigest()
    conn = _connect()
    try:
        existing = conn.execute("SELECT id FROM documents WHERE sha256=?", (digest,)).fetchone()
        if existing:
            return 0, "Tài liệu này đã tồn tại, bỏ qua"
        pages = _extract_text(filename, raw)
        chunks = _chunk_pages(pages)
        if not chunks:
            raise ValueError("Không trích xuất được văn bản. Nếu đây là PDF scan, hãy OCR trước.")
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), 48):
            vectors.extend(_embed([c[1] for c in chunks[start:start + 48]], base_url, embed_model))
        if len(vectors) != len(chunks):
            raise OllamaError("Số embedding không khớp số đoạn văn bản.")
        cursor = conn.execute(
            "INSERT INTO documents(name,sha256,created_at,chunks) VALUES(?,?,?,?)",
            (Path(filename).name, digest, datetime.now().strftime("%Y-%m-%d %H:%M"), len(chunks)),
        )
        doc_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO chunks(document_id,chunk_index,location,content,embedding) VALUES(?,?,?,?,?)",
            [(doc_id, i, chunks[i][0], chunks[i][1], json.dumps(vectors[i])) for i in range(len(chunks))],
        )
        conn.commit()
        return len(chunks), "Đã nạp thành công"
    finally:
        conn.close()


def list_documents() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT id,name,created_at,chunks FROM documents ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_document(document_id: int) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
    finally:
        conn.close()


def retrieve(query: str, base_url: str, embed_model: str, top_k: int = 4) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("""SELECT c.content,c.location,c.embedding,d.name
            FROM chunks c JOIN documents d ON d.id=c.document_id""").fetchall()
    finally:
        conn.close()
    if not rows:
        return []

    query_vector = np.asarray(_embed([query], base_url, embed_model)[0], dtype=np.float32)
    matrix = np.asarray([json.loads(row["embedding"]) for row in rows], dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominators = row_norms * query_norm
    scores = np.divide(
        matrix @ query_vector, denominators,
        out=np.zeros(len(rows), dtype=np.float32), where=denominators > 0,
    )
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": rows[int(i)]["content"], "location": rows[int(i)]["location"],
         "name": rows[int(i)]["name"], "score": float(scores[int(i)])}
        for i in best_indices
    ]
