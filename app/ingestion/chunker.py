from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loader import DocumentPage


@dataclass
class DocumentChunk:
    """A single text chunk derived from a DocumentPage, ready for embedding."""

    chunk_id: str
    text: str
    doc_name: str
    page_number: int   # 0-indexed
    page_label: int    # 1-indexed (human-readable)
    total_pages: int
    upload_timestamp: str
    source_path: str
    chunk_index: int   # position within the page's chunks

    def to_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "page_label": self.page_label,
            "total_pages": self.total_pages,
            "upload_timestamp": self.upload_timestamp,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
        }


class Chunker:
    """Split DocumentPage objects into overlapping text chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Split a list of DocumentPage objects into DocumentChunk objects.

        Args:
            pages: Pages returned by PDFLoader.

        Returns:
            List of DocumentChunk objects with full provenance metadata.
        """
        chunks: List[DocumentChunk] = []

        for page in pages:
            raw_chunks = self._splitter.split_text(page.text)
            for idx, text in enumerate(raw_chunks):
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=text,
                        doc_name=page.doc_name,
                        page_number=page.page_number,
                        page_label=page.page_label,
                        total_pages=page.total_pages,
                        upload_timestamp=page.upload_timestamp,
                        source_path=page.source_path,
                        chunk_index=idx,
                    )
                )

        return chunks
