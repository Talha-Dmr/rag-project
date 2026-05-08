"""
Section-aware chunking strategy.

Splits text along lightweight document structure signals such as page markers,
numbered headings, and short heading-like lines, while preserving section/page
metadata on each chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from src.core.base_classes import BaseChunker
from src.chunking.base_chunker import register_chunker
from src.core.logger import get_logger

logger = get_logger(__name__)


PAGE_MARKER_RE = re.compile(r"^\[Page\s+(\d+)\]$", re.IGNORECASE)
NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}|[A-Z])(?:[.)])?\s+[A-Z][A-Za-z0-9/\-(),:& ]{2,120}$"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _SectionBlock:
    heading: str
    page: Optional[int]
    text: str


@register_chunker("section_aware")
class SectionAwareChunker(BaseChunker):
    """
    Chunk text using lightweight document structure.

    This strategy is intentionally simple: it keeps page markers, short numbered
    headings, and short heading-like lines together as section boundaries, then
    packs section blocks into chunks with page/section metadata preserved.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_chunk_size = int(self.config.get("max_chunk_size", 1200))
        self.min_chunk_size = int(self.config.get("min_chunk_size", 200))
        self.chunk_overlap = int(self.config.get("chunk_overlap", 120))
        self.max_heading_length = int(self.config.get("max_heading_length", 120))
        self.allow_titlecase_headings = bool(self.config.get("allow_titlecase_headings", True))

        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")

        logger.info(
            "Initialized SectionAwareChunker: max_size=%s, min_size=%s, overlap=%s",
            self.max_chunk_size,
            self.min_chunk_size,
            self.chunk_overlap,
        )

    def chunk(self, text: str) -> List[str]:
        chunk_docs = self._chunk_text_with_metadata(text)
        return [doc["content"] for doc in chunk_docs]

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_docs: List[Dict[str, Any]] = []

        for doc_idx, document in enumerate(documents):
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            chunks = self._chunk_text_with_metadata(content)

            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append(
                    {
                        "content": chunk["content"],
                        "metadata": {
                            **metadata,
                            **chunk["metadata"],
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "chunking_strategy": "section_aware",
                            "max_chunk_size": self.max_chunk_size,
                            "min_chunk_size": self.min_chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "original_doc_index": doc_idx,
                        },
                    }
                )

        logger.info(
            "Section-aware chunked %s documents into %s chunks",
            len(documents),
            len(chunked_docs),
        )
        return chunked_docs

    def _chunk_text_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        blocks = self._extract_section_blocks(text)
        if not blocks:
            blocks = [_SectionBlock(heading="", page=None, text=text.strip())]

        chunks: List[Dict[str, Any]] = []
        current_parts: List[str] = []
        current_heading: Optional[str] = None
        current_page_start: Optional[int] = None
        current_page_end: Optional[int] = None
        current_len = 0

        for block in blocks:
            block_heading = block.heading.strip() if block.heading else ""
            for block_rendered in self._render_block_parts(block_heading, block.text.strip()):
                if not block_rendered:
                    continue

                block_len = len(block_rendered)
                would_exceed = current_parts and (current_len + block_len > self.max_chunk_size)

                if would_exceed:
                    chunks.append(
                        self._make_chunk(
                            current_parts,
                            current_heading,
                            current_page_start,
                            current_page_end,
                        )
                    )
                    current_parts = self._build_overlap_parts(current_parts)
                    current_len = sum(len(part) for part in current_parts)
                    if not current_parts:
                        current_heading = None
                        current_page_start = None
                        current_page_end = None

                if not current_parts:
                    current_heading = block_heading or current_heading or ""
                    current_page_start = block.page
                current_page_end = block.page if block.page is not None else current_page_end

                current_parts.append(block_rendered)
                current_len += block_len

                if current_len >= self.max_chunk_size:
                    chunks.append(
                        self._make_chunk(
                            current_parts,
                            current_heading,
                            current_page_start,
                            current_page_end,
                        )
                    )
                    current_parts = self._build_overlap_parts(current_parts)
                    current_len = sum(len(part) for part in current_parts)
                    if not current_parts:
                        current_heading = None
                        current_page_start = None
                        current_page_end = None

        if current_parts:
            chunks.append(
                self._make_chunk(
                    current_parts,
                    current_heading,
                    current_page_start,
                    current_page_end,
                )
            )

        return chunks

    def _extract_section_blocks(self, text: str) -> List[_SectionBlock]:
        blocks: List[_SectionBlock] = []
        current_page: Optional[int] = None
        current_heading: str = ""
        paragraph_lines: List[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if not paragraph_lines:
                return
            paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
            paragraph_lines = []
            if paragraph:
                blocks.append(_SectionBlock(heading=current_heading, page=current_page, text=paragraph))

        lines = [line.rstrip() for line in text.splitlines()]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue

            page_match = PAGE_MARKER_RE.match(stripped)
            if page_match:
                flush_paragraph()
                current_page = int(page_match.group(1))
                continue

            if self._is_heading(stripped):
                flush_paragraph()
                current_heading = stripped
                continue

            paragraph_lines.append(stripped)

        flush_paragraph()
        return blocks

    def _is_heading(self, line: str) -> bool:
        if len(line) > self.max_heading_length:
            return False
        if line.endswith((".", "?", "!", ";", ":")):
            return False
        if NUMBERED_HEADING_RE.match(line):
            return True
        if not self.allow_titlecase_headings:
            return False
        words = line.split()
        if len(words) < 2 or len(words) > 12:
            return False
        if any(word.isdigit() for word in words):
            return False
        capitalized = sum(1 for word in words if word[:1].isupper())
        return capitalized >= max(2, len(words) - 1)

    def _build_overlap_parts(self, parts: List[str]) -> List[str]:
        if not parts or self.chunk_overlap <= 0:
            return []

        tail = parts[-1].strip()
        if len(tail) <= self.chunk_overlap:
            return [tail]

        window_size = min(len(tail), self.chunk_overlap * 3)
        window = tail[-window_size:].strip()
        sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(window) if sentence.strip()]

        overlap = ""
        for sentence in reversed(sentences):
            candidate = f"{sentence} {overlap}".strip() if overlap else sentence
            if len(candidate) > self.chunk_overlap:
                break
            overlap = candidate

        if overlap and len(overlap) >= min(40, self.chunk_overlap):
            return [overlap]

        return []

    def _render_block_parts(self, heading: str, text: str) -> List[str]:
        if not text:
            return []

        rendered = f"{heading}\n{text}" if heading else text
        if len(rendered) <= self.max_chunk_size:
            return [rendered]

        max_body_len = max(200, self.max_chunk_size - len(heading) - 1 if heading else self.max_chunk_size)
        body_parts = self._split_long_text(text, max_body_len)
        return [f"{heading}\n{part}" if heading else part for part in body_parts]

    def _split_long_text(self, text: str, max_len: int) -> List[str]:
        sentences = [s.strip() for s in SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
        if len(sentences) <= 1:
            return [text[i:i + max_len].strip() for i in range(0, len(text), max_len) if text[i:i + max_len].strip()]

        parts: List[str] = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            candidate = f"{current} {sentence}"
            if len(candidate) <= max_len:
                current = candidate
            else:
                parts.append(current)
                current = sentence
        if current:
            parts.append(current)
        return parts

    def _make_chunk(
        self,
        parts: List[str],
        heading: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> Dict[str, Any]:
        content = "\n\n".join(part for part in parts if part.strip()).strip()
        metadata: Dict[str, Any] = {}
        if heading:
            metadata["section_heading"] = heading
        if page_start is not None:
            metadata["page_start"] = page_start
        if page_end is not None:
            metadata["page_end"] = page_end
        return {"content": content, "metadata": metadata}
