from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class DocumentPage:
    """Single page extracted from a PDF document."""

    text: str
    doc_name: str
    page_number: int  # 0-indexed as stored by PyMuPDF, exposed 1-indexed via property
    total_pages: int
    upload_timestamp: str
    source_path: str
    metadata: dict = field(default_factory=dict)

    @property
    def page_label(self) -> int:
        """Human-readable 1-indexed page number."""
        return self.page_number + 1


class PDFLoader:
    """Load PDF documents using PyMuPDF and return structured page objects."""

    def load_file(self, file_path: str | Path) -> List[DocumentPage]:
        """
        Load a single PDF file and return one DocumentPage per page.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of DocumentPage objects, one per page.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        doc_name = path.stem
        upload_timestamp = datetime.now(timezone.utc).isoformat()
        pages: List[DocumentPage] = []

        with fitz.open(str(path)) as doc:
            total = len(doc)
            for page_index in range(total):
                page = doc[page_index]
                text = page.get_text("text")
                if not text.strip():
                    continue  # skip blank pages
                pages.append(
                    DocumentPage(
                        text=text,
                        doc_name=doc_name,
                        page_number=page_index,
                        total_pages=total,
                        upload_timestamp=upload_timestamp,
                        source_path=str(path),
                    )
                )

        return pages

    def load_directory(self, directory: str | Path) -> List[DocumentPage]:
        """
        Recursively load all PDF files from a directory.

        Args:
            directory: Path to the directory.

        Returns:
            Flat list of DocumentPage objects from all PDFs found.
        """
        directory = Path(directory)
        all_pages: List[DocumentPage] = []
        pdf_files = sorted(directory.rglob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {directory}")

        for pdf_path in pdf_files:
            pages = self.load_file(pdf_path)
            all_pages.extend(pages)

        return all_pages
