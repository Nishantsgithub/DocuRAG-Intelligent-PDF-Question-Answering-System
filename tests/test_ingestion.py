"""Tests for the ingestion module (loader + chunker)."""
from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunker, DocumentChunk
from app.ingestion.loader import DocumentPage, PDFLoader


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_page(text: str = "Hello world. " * 50, page_number: int = 0) -> DocumentPage:
    return DocumentPage(
        text=text,
        doc_name="test_doc",
        page_number=page_number,
        total_pages=1,
        upload_timestamp="2024-01-01T00:00:00+00:00",
        source_path="test_doc.pdf",
    )


# ── PDFLoader ──────────────────────────────────────────────────────────────────

class TestPDFLoader:
    def test_raises_for_missing_file(self, tmp_path):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "nonexistent.pdf")

    def test_raises_for_non_pdf(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")
        loader = PDFLoader()
        with pytest.raises(ValueError, match=r"\.pdf"):
            loader.load_file(txt)

    def test_raises_for_empty_directory(self, tmp_path):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory(tmp_path)

    def test_load_file_returns_document_pages(self, tmp_path):
        """Create a minimal valid PDF and verify loader returns DocumentPage objects."""
        pytest.importorskip("fitz")

        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Sample text for testing.", fontsize=12)
        pdf_path = tmp_path / "sample.pdf"
        doc.save(str(pdf_path))
        doc.close()

        loader = PDFLoader()
        pages = loader.load_file(pdf_path)

        assert len(pages) >= 1
        assert all(isinstance(p, DocumentPage) for p in pages)
        assert pages[0].doc_name == "sample"
        assert pages[0].page_label == 1
        assert "Sample text" in pages[0].text


# ── Chunker ────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_produces_document_chunks(self):
        page = _make_page()
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_pages([page])

        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_metadata_propagation(self):
        page = _make_page(page_number=3)
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_pages([page])

        for chunk in chunks:
            assert chunk.doc_name == "test_doc"
            assert chunk.page_number == 3
            assert chunk.page_label == 4  # 0-indexed + 1

    def test_chunk_ids_are_unique(self):
        pages = [_make_page(page_number=i) for i in range(3)]
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_pages(pages)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_coverage(self):
        """All original words should appear in at least one chunk."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        page = _make_page(text=text)
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_pages([page])

        combined = " ".join(c.text for c in chunks)
        assert "quick brown fox" in combined

    def test_to_metadata_keys(self):
        page = _make_page()
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        chunk = chunker.chunk_pages([page])[0]
        meta = chunk.to_metadata()

        for key in ("chunk_id", "doc_name", "page_number", "page_label", "upload_timestamp"):
            assert key in meta
