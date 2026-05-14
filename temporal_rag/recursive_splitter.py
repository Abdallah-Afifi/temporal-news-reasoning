"""
Split long articles into smaller overlapping chunks using token counts.


We avoid blind character cuts. The splitter tries paragraph gaps first, then sentences,
then words, until each chunk fits the token limit. Neighboring chunks share a short
overlap so context is not lost at boundaries.

Temporal RAG System · Detailed Architecture v4
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenEstimationMethod(Enum):
    """Methods for estimating token count without tokenization."""
    WORD_RATIO = "word_ratio"          # ~1.3 tokens per word (English avg)
    CHARACTER_RATIO = "character_ratio" # ~4 characters per token (English avg)
    WHITESPACE = "whitespace"           # Split on whitespace, simple count


@dataclass
class ChunkMetadata:
    """Metadata associated with a text chunk."""
    chunk_id: int
    text: str
    token_count: int
    character_count: int
    start_char_index: int
    end_char_index: int
    source_doc_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0

    # overlap metadata (measured from final stored indices)
    overlap_tokens: int = 0
    overlap_start_char_index: int = 0
    overlap_end_char_index: int = 0
    title: str = ""
    published_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'token_count': self.token_count,
            'character_count': self.character_count,
            'start_char_index': self.start_char_index,
            'end_char_index': self.end_char_index,
            'source_doc_id': self.source_doc_id,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'overlap_tokens': self.overlap_tokens,
            'overlap_start_char_index': self.overlap_start_char_index,
            'overlap_end_char_index': self.overlap_end_char_index,
            'title': self.title,
            'published_date': self.published_date,
        }


class TokenCounter:
    """
    Real tokenizer using tiktoken (OpenAI-compatible).
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        # Avoid network/model-resolution stalls during startup.
        # cl100k_base is compatible with modern embedding/chat models.
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = tiktoken.encoding_for_model(model_name)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))


class RecursiveCharacterSplitter:
    """
    Recursively splits text into chunks while preserving semantic boundaries.
    
    The splitter works by:
    1. Attempting to split by the primary separator (usually paragraphs)
    2. If chunks are too large, recursively split by secondary separators
    3. Cascades through separators: paragraph → sentence → word → character
    4. Maintains overlap between chunks for context preservation
    
    Configuration:
        chunk_tokens: Target chunk size in tokens (default: 512)
        overlap_tokens: Overlap size in tokens (default: 50)
        
    LIMITATIONS:
    - Doesn't preserve code block formatting
    - May break on unusual whitespace or control characters
    - Separator-aware splitting may not work for all languages
    """
    
    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        token_counter: Optional[TokenCounter] = None,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        Initialize the recursive character splitter.
        
        Args:
            chunk_tokens: Target chunk size in tokens (default: 512)
            overlap_tokens: Overlap between chunks in tokens (default: 50)
            token_counter: TokenCounter instance (creates default if None)
            separators: List of separators to try in order (see defaults below)
            keep_separator: Whether to include separator in chunks
            
        Default separators (cascade order):
            - "\n\n" (paragraph break)
            - "\n" (line break)
            - ". " (sentence end)
            - " " (word break)
            - "" (character level - last resort)
        """
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = token_counter or TokenCounter()
        self.keep_separator = keep_separator
        
        # Default separators cascade through semantic levels
        self.separators = separators or [
            "\n\n",  # paragraph break
            "\n",    # line break
            ". ",    # sentence end
            "? ",    # sentence end (question)
            "! ",    # sentence end (exclamation)
            " ",     # word break
            ""       # character level
        ]
        
        self._chunk_id_counter = 0
    
    def split(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        title: str = "",
        published_date: Optional[str] = None,
    ) -> List[ChunkMetadata]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text to split.
            source_doc_id: Optional identifier for the source document (e.g. ``"doc_0"``).
            title: Article headline, stored in each chunk's metadata.
            published_date: Publication date string (``YYYY-MM-DD``), stored in each chunk's metadata.

        Returns:
            List of ChunkMetadata objects
            
        Process:
            1. Recursively splits by separators
            2. Applies overlap between chunks
            3. Merges very small tail chunks
            4. Enforces strict size limits
            5. Attaches metadata with indices and token counts
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return []

        # Keep prior sanitation behavior.
        text = self._sanitize_text_for_chunking(text)

        token_ids = self.token_counter.encoding.encode(text)
        if not token_ids:
            return []

        # Cache decoded prefix lengths to map token indices back to char indices.
        prefix_char_cache: Dict[int, int] = {0: 0}

        def prefix_char_pos(tok_idx: int) -> int:
            tok_idx = max(0, min(tok_idx, len(token_ids)))
            if tok_idx not in prefix_char_cache:
                prefix_char_cache[tok_idx] = len(self.token_counter.encoding.decode(token_ids[:tok_idx]))
            return prefix_char_cache[tok_idx]

        spans: List[Tuple[int, int]] = []
        start_tok = 0
        while start_tok < len(token_ids):
            hard_end = min(start_tok + self.chunk_tokens, len(token_ids))
            end_tok = self._choose_end_token_boundary(token_ids, start_tok, hard_end)
            if end_tok <= start_tok:
                end_tok = hard_end

            spans.append((start_tok, end_tok))
            if end_tok >= len(token_ids):
                break

            desired_start = max(start_tok + 1, end_tok - self.overlap_tokens)
            next_start = self._adjust_start_token_boundary(token_ids, desired_start, end_tok)
            # Ensure strict progress and overlap upper bound.
            next_start = max(start_tok + 1, min(next_start, end_tok))
            if end_tok - next_start > self.overlap_tokens:
                next_start = end_tok - self.overlap_tokens
            start_tok = next_start

        chunks: List[ChunkMetadata] = []
        for idx, (s_tok, e_tok) in enumerate(spans):
            chunk_text = self.token_counter.encoding.decode(token_ids[s_tok:e_tok])
            if not chunk_text.strip():
                continue
            start_char = prefix_char_pos(s_tok)
            end_char = prefix_char_pos(e_tok)
            token_count = e_tok - s_tok

            cm = ChunkMetadata(
                chunk_id=self._chunk_id_counter,
                text=chunk_text,
                token_count=token_count,
                character_count=len(chunk_text),
                start_char_index=start_char,
                end_char_index=end_char,
                source_doc_id=source_doc_id,
                chunk_index=idx,
                total_chunks=len(spans),
                title=title,
                published_date=published_date,
            )
            self._chunk_id_counter += 1

            if chunks:
                prev = chunks[-1]
                prev_s_tok, prev_e_tok = spans[idx - 1]
                overlap_tok_count = max(0, prev_e_tok - s_tok)
                cm.overlap_tokens = min(overlap_tok_count, self.overlap_tokens)
                cm.overlap_start_char_index = prefix_char_pos(s_tok)
                cm.overlap_end_char_index = prefix_char_pos(prev_e_tok)

            chunks.append(cm)

        for i, cm in enumerate(chunks):
            cm.chunk_index = i
            cm.total_chunks = len(chunks)

        return chunks

    def _boundary_score(self, token_ids: List[int], boundary_idx: int) -> int:
        """Score how good a token boundary is for splitting.

        Higher is better.  Scores:
          5 : sentence ending (``"."`` / ``"!"`` / ``"?"`` before whitespace)
          4 : double newline (paragraph break)
          3 : single newline (line break)
          2 : whitespace (word boundary)
          1 : other safe boundary
         -1 : mid-word (both sides are alphanumeric) - avoid this

        The chunk end-finder uses this to pick the cleanest cut point within a
        search window near the token limit.
        """
        left = self.token_counter.encoding.decode(token_ids[max(0, boundary_idx - 4):boundary_idx])
        right = self.token_counter.encoding.decode(token_ids[boundary_idx:min(len(token_ids), boundary_idx + 4)])

        left_last = left[-1] if left else ""
        right_first = right[0] if right else ""
        left_stripped = left.rstrip()
        left_tail = left_stripped[-2:] if len(left_stripped) >= 2 else left_stripped

        if (left_last.isalnum() and right_first.isalnum()):
            return -1
        if left_tail.endswith("\n\n") or right.startswith("\n\n"):
            return 4
        if left_stripped.endswith((".", "!", "?")):
            return 5
        if left_last == "\n" or right_first == "\n":
            return 3
        if left_last.isspace() or right_first.isspace():
            return 2
        return 1

    def _choose_end_token_boundary(self, token_ids: List[int], start_tok: int, hard_end: int) -> int:
        """Return the best token index to end a chunk, staying at or before ``hard_end``.

        Searches backward up to 120 tokens from ``hard_end`` to find the highest-scoring
        boundary (sentence end > paragraph > newline > word).  If every candidate scores
        as a mid-word cut (score −1), a last safety pass finds the closest non-mid-word
        position.  Falls back to ``hard_end`` if nothing better is found.
        """
        if hard_end >= len(token_ids):
            return hard_end

        search_floor = max(start_tok + 1, hard_end - 120)
        best_idx = hard_end
        best_score = self._boundary_score(token_ids, hard_end)

        for cand in range(hard_end, search_floor - 1, -1):
            score = self._boundary_score(token_ids, cand)
            if score > best_score:
                best_score = score
                best_idx = cand
                if best_score >= 5:
                    break

        if best_score < 0:
            # Last safety pass: avoid hard mid-word split when possible.
            for cand in range(hard_end - 1, search_floor - 1, -1):
                if self._boundary_score(token_ids, cand) >= 0:
                    return cand
            return hard_end
        return best_idx

    def _adjust_start_token_boundary(
        self,
        token_ids: List[int],
        desired_start: int,
        current_end: int,
    ) -> int:
        """Nudge ``desired_start`` forward (up to 20 tokens) to land on a clean boundary.

        When the overlap window starts mid-word, this pushes the start position
        just past the next whitespace or punctuation so the chunk does not begin
        with a partial word.  Returns ``desired_start`` unchanged if no clean
        boundary is found within the search window.
        """
        if desired_start >= current_end:
            return current_end

        max_forward = min(current_end - 1, desired_start + 20)
        for cand in range(desired_start, max_forward + 1):
            if self._boundary_score(token_ids, cand) >= 0:
                return cand
        return desired_start
    
    def _enforce_size_limits(
        self,
        chunks: List[ChunkMetadata],
        original_text: str,
        hard_limit: Optional[int] = None
    ) -> List[ChunkMetadata]:
        """
        Enforce strict size limits on chunks.
        
        Any chunk exceeding hard_limit will be re-split by the splitter.
        """
        if hard_limit is None:
            hard_limit = self.chunk_tokens
        
        compliant = []
        
        for chunk in chunks:
            if chunk.token_count > hard_limit:
                logger.warning(
                    f"Chunk {chunk.chunk_id} exceeds size limit: "
                    f"{chunk.token_count} > {hard_limit} tokens. Re-splitting..."
                )
                
                chunk_text = original_text[chunk.start_char_index:chunk.end_char_index]
                sub_chunks = self._recursive_split(chunk_text)
                
                for sub_chunk_text in sub_chunks:
                    local_pos = chunk_text.find(sub_chunk_text)
                    if local_pos >= 0:
                        global_start = chunk.start_char_index + local_pos
                        global_end = global_start + len(sub_chunk_text)
                        
                        stripped_text = sub_chunk_text.strip()
                        if stripped_text:
                            sub_token_count = self.token_counter.estimate_tokens(stripped_text)
                            meta = ChunkMetadata(
                                chunk_id=self._chunk_id_counter,
                                text=stripped_text,
                                token_count=sub_token_count,
                                character_count=len(stripped_text),
                                start_char_index=global_start,
                                end_char_index=global_end,
                                source_doc_id=chunk.source_doc_id,
                                chunk_index=len(compliant),
                                total_chunks=len(chunks)
                            )
                            self._chunk_id_counter += 1
                            compliant.append(meta)
            else:
                compliant.append(chunk)
        
        for i, chunk in enumerate(compliant):
            chunk.chunk_index = i
            chunk.total_chunks = len(compliant)
        
        return compliant
    
    def _merge_small_tail_chunks(
        self,
        chunks: List[ChunkMetadata],
        original_text: str,
        tail_threshold: int = 200
    ) -> List[ChunkMetadata]:
        """
        Merge very small chunks into their predecessor.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        for current_chunk in chunks:
            if current_chunk.token_count <= tail_threshold and merged:
                prev_chunk = merged[-1]
                if current_chunk.start_char_index >= prev_chunk.end_char_index:
                    merged_text = original_text[prev_chunk.start_char_index:current_chunk.end_char_index]
                    merged_tokens = self.token_counter.estimate_tokens(merged_text)
                    if merged_tokens <= int(self.chunk_tokens * 1.1):
                        merged[-1] = ChunkMetadata(
                            chunk_id=prev_chunk.chunk_id,
                            text=merged_text,
                            token_count=merged_tokens,
                            character_count=len(merged_text),
                            start_char_index=prev_chunk.start_char_index,
                            end_char_index=current_chunk.end_char_index,
                            source_doc_id=prev_chunk.source_doc_id,
                            chunk_index=prev_chunk.chunk_index,
                            total_chunks=len(chunks) - 1,
                        )
                        continue
            merged.append(current_chunk)
        
        for idx, chunk in enumerate(merged):
            chunk.chunk_index = idx
            chunk.total_chunks = len(merged)
        
        return merged

    def _merge_small_raw_chunks(
        self,
        chunks: List[str],
        min_tokens: int = 200,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Merge undersized non-overlapping raw chunks with neighbors.
        """
        if len(chunks) <= 1:
            return chunks

        max_tokens = max_tokens or int(self.chunk_tokens * 1.2)
        merged = list(chunks)
        i = 0
        while i < len(merged):
            current_tokens = self.token_counter.estimate_tokens(merged[i])
            if current_tokens >= min_tokens:
                i += 1
                continue

            merged_into_neighbor = False
            if i > 0:
                left_text = merged[i - 1] + merged[i]
                if self.token_counter.estimate_tokens(left_text) <= max_tokens:
                    merged[i - 1] = left_text
                    del merged[i]
                    i = max(0, i - 1)
                    merged_into_neighbor = True

            if not merged_into_neighbor and i < len(merged) - 1:
                right_text = merged[i] + merged[i + 1]
                if self.token_counter.estimate_tokens(right_text) <= max_tokens:
                    merged[i] = right_text
                    del merged[i + 1]
                    merged_into_neighbor = True

            if not merged_into_neighbor and i == len(merged) - 1 and i > 0:
                tail_text = merged[i - 1] + merged[i]
                if self.token_counter.estimate_tokens(tail_text) <= int(self.chunk_tokens * 1.4):
                    merged[i - 1] = tail_text
                    del merged[i]
                    i = max(0, i - 1)
                    merged_into_neighbor = True

            if not merged_into_neighbor:
                i += 1

        return merged

    def _recursive_split(
        self,
        text: str,
        separator_index: int = 0
    ) -> List[str]:
        """
        Recursively split text by cascade of separators.
        """
        separator = self.separators[separator_index]
        chunks = []
        
        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)
        
        if separator and self.keep_separator and separator_index < len(self.separators) - 1:
            good_splits = []
            for part in parts:
                if part:
                    good_splits.append(part + separator)
            if parts and parts[-1]:
                good_splits[-1] = good_splits[-1][:-len(separator)]
            parts = good_splits
        else:
            parts = [p for p in parts if p]
        
        good_splits = []
        bad_splits = []
        
        for part in parts:
            token_count = self.token_counter.estimate_tokens(part)
            if token_count <= self.chunk_tokens:
                good_splits.append(part)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    chunks.extend(merged)
                    good_splits = []
                
                if separator_index < len(self.separators) - 1:
                    sub_chunks = self._recursive_split(part, separator_index + 1)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(part)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            chunks.extend(merged)
        
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self.token_counter.estimate_tokens(chunk)
            if chunk_tokens > self.chunk_tokens and separator_index < len(self.separators) - 1:
                sub_chunks = self._recursive_split(chunk, separator_index + 1)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge small splits together without exceeding chunk size limit.
        """
        separator_tokens = self.token_counter.estimate_tokens(separator)
        merged = []
        current = ""
        current_tokens = 0
        
        soft_limit = int(self.chunk_tokens * 0.9)
        hard_limit = self.chunk_tokens
        
        for split in splits:
            split_tokens = self.token_counter.estimate_tokens(split)
            
            if split_tokens > hard_limit:
                if current:
                    merged.append(current)
                    current = ""
                    current_tokens = 0
                merged.append(split)
                continue
            
            if current:
                proposed_tokens = current_tokens + separator_tokens + split_tokens
            else:
                proposed_tokens = split_tokens
            
            if proposed_tokens <= soft_limit:
                if current:
                    current += separator + split
                    current_tokens = proposed_tokens
                else:
                    current = split
                    current_tokens = split_tokens
            elif proposed_tokens <= hard_limit:
                if current:
                    current += separator + split
                    current_tokens = proposed_tokens
                else:
                    current = split
                    current_tokens = split_tokens
                merged.append(current)
                current = ""
                current_tokens = 0
            else:
                if current:
                    merged.append(current)
                current = split
                current_tokens = split_tokens
        
        if current:
            merged.append(current)
        
        return merged
    
    def _snap_to_boundary(self, text: str, pos: int, direction: str = 'forward') -> int:
        """
        Snap a position to the nearest safe boundary, preferring sentence endings.
        """
        if pos <= 0:
            return 0
        if pos >= len(text):
            return len(text)
        
        sentence_endings = {'. ', '! ', '? '}
        other_boundaries = {' ', '\n', '\t'}
        
        if direction == 'backward':
            for i in range(pos - 2, max(0, pos - 300), -1):
                if i < len(text) - 1 and text[i:i+2] in sentence_endings:
                    return i + 2
            
            for i in range(pos - 1, -1, -1):
                if i < 0:
                    return 0
                if text[i] in other_boundaries:
                    return i + 1
            
            return 0
        else:  # forward
            skip_pos = pos
            while skip_pos < len(text) and text[skip_pos].isspace():
                skip_pos += 1

            if skip_pos >= len(text):
                return len(text)

            if skip_pos > 0 and text[skip_pos - 1].isalnum() and text[skip_pos].isalnum():
                boundary_chars = set(' \n\t.!?,;:')
                next_pos = skip_pos
                while next_pos < len(text) and text[next_pos] not in boundary_chars:
                    next_pos += 1
                if next_pos < len(text) and text[next_pos] in {'.', '!', '?'} and next_pos + 1 < len(text) and text[next_pos+1] == ' ':
                    return next_pos + 2
                else:
                    return next_pos

            return skip_pos

    def _sanitize_text_for_chunking(self, text: str) -> str:
        """Mask common extraction artifacts with whitespace of equal length."""
        if not text:
            return text

        patterns = [
            r"Read:\s*\.",
            r"Read:\s*$",
            r"\u200b",
            r"\s+\.\s*$",
        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, lambda m: " " * len(m.group(0)), cleaned, flags=re.MULTILINE)
        return cleaned

    def _clean_chunk_start(self, text: str) -> str:
        """
        Remove leading punctuation artifacts from chunk text that can appear
        when the overlap boundary lands just after a sentence-ending period.

        Examples of artifacts cleaned:
            ". Now she runs..."  ->  "Now she runs..."
            ", and the story..." ->  "and the story..."  (only if leading punct)
        """
        # Strip leading whitespace first
        text = text.lstrip()

        # Remove a lone leading punctuation character followed by whitespace
        # that has no business starting a sentence (period, comma, semicolon).
        # We are conservative: only match a single char so we don't eat real content.
        text = re.sub(r'^([.,;])\s+', '', text)

        return text

    def _prefer_sentence_boundary_start(
        self,
        text: str,
        overlap_start: int,
        base_start: int,
        min_overlap_tokens: Optional[int] = None
    ) -> int:
        """
        Move overlap start to a sentence boundary when it still preserves context.
        """
        if base_start <= overlap_start:
            return overlap_start
        base_start = min(base_start, len(text))

        if min_overlap_tokens is None:
            min_overlap_tokens = max(10, self.overlap_tokens // 3)
        max_overlap_tokens = self.overlap_tokens + 30

        candidate = overlap_start
        found_backward_boundary = False
        search_floor = max(0, overlap_start - 250)
        for i in range(base_start - 2, search_floor - 1, -1):
            if text[i] in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
                candidate = i + 2
                found_backward_boundary = True
                break

        if not found_backward_boundary:
            for i in range(overlap_start, base_start - 1):
                if text[i] in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
                    candidate = i + 2
                    break

        if candidate <= overlap_start or candidate >= base_start:
            return overlap_start
        overlap_text = text[candidate:base_start]
        overlap_tokens = self.token_counter.estimate_tokens(overlap_text)
        if overlap_tokens < min_overlap_tokens or overlap_tokens > max_overlap_tokens:
            return overlap_start
        return candidate
    
    def _fix_weak_ending(self, chunk_text: str, full_text: str) -> str:
        """Extend a chunk if it ends on a weak or awkward boundary."""
        stripped = chunk_text.rstrip()
        if stripped.endswith(('.', '!', '?')):
            return chunk_text
        if stripped.endswith((':', ';')):
            idx = full_text.find(chunk_text)
            if idx == -1:
                return chunk_text
            end_pos = idx + len(chunk_text)
            new_end = self._snap_to_boundary(full_text, end_pos, direction='forward')
            if new_end > end_pos:
                return full_text[idx:new_end].strip()
        return chunk_text
    
    def _apply_overlap_and_metadata(
        self,
        chunks: List[str],
        original_text: str,
        source_doc_id: Optional[str],
        title: str = "",
        published_date: Optional[str] = None,
    ) -> List[ChunkMetadata]:
        """
        Apply overlap between chunks and attach metadata.

        Args:
            chunks: List of text chunks from recursive split (non-overlapping)
            original_text: Original full text (for position tracking)
            source_doc_id: Source document ID

        Returns:
            List of ChunkMetadata objects with overlap applied

        Overlap mechanism:
            - For each chunk after the first, extend backward to include 50-token overlap
            - Overlap is the shared content between consecutive chunks
            - Uses token-based calculation, not character estimates
        """
        if not chunks:
            return []

        # Rebuild positions in original text
        chunks_with_positions = self._find_chunk_positions(chunks, original_text)

        # Build metadata with overlap
        overlapped_chunks = []
        prev_end_pos = None

        for i, (chunk, start_pos, end_pos) in enumerate(chunks_with_positions):

            # Snap end position to boundary (preferring sentence endings), but
            # keep the last chunk's full tail to avoid coverage gaps.
            if i == len(chunks_with_positions) - 1:
                snapped_end = end_pos
            else:
                snapped_end = self._snap_to_boundary(original_text, end_pos, direction='backward')

            # Determine chunk start position (with overlap for non-first chunks)
            chunk_start = start_pos
            if i > 0 and overlapped_chunks:
                prev_meta = overlapped_chunks[-1]

                # Anchor overlap to the END of the previous stored chunk
                anchor_end = prev_meta.end_char_index
                target_overlap = self.overlap_tokens

                # Search back far enough (500 chars is often not enough for 50 tokens)
                max_back = max(800, target_overlap * 40)  # e.g., 50*40=2000 chars
                tentative_start = max(0, anchor_end - max_back)

                best_pos = anchor_end
                best_tokens = 0

                # Walk backward to find earliest position still <= target_overlap tokens
                for candidate_pos in range(anchor_end - 1, tentative_start - 1, -1):
                    overlap_text = original_text[candidate_pos:anchor_end]
                    overlap_tokens = self.token_counter.estimate_tokens(overlap_text)

                    if overlap_tokens <= target_overlap:
                        best_pos = candidate_pos
                        best_tokens = overlap_tokens
                        continue
                    else:
                        break

                # One more expansion if we ended up way under target
                if best_tokens < target_overlap - 5 and tentative_start > 0:
                    tentative_start2 = max(0, anchor_end - max_back * 2)
                    for candidate_pos in range(anchor_end - 1, tentative_start2 - 1, -1):
                        overlap_text = original_text[candidate_pos:anchor_end]
                        overlap_tokens = self.token_counter.estimate_tokens(overlap_text)

                        if overlap_tokens <= target_overlap:
                            best_pos = candidate_pos
                            best_tokens = overlap_tokens
                            continue
                        else:
                            break

                chunk_start = best_pos

            # Snap the chunk start to a word boundary
            forward_start = self._snap_to_boundary(original_text, chunk_start, direction="forward")

            if overlapped_chunks:
                prev_meta = overlapped_chunks[-1]
                overlap_text = original_text[forward_start:prev_meta.end_char_index]
                overlap_tokens = self.token_counter.estimate_tokens(overlap_text)
                if overlap_tokens >= max(10, self.overlap_tokens - 10):
                    chunk_start = forward_start
            else:
                chunk_start = forward_start

            # Extract chunk text
            chunk_text = original_text[chunk_start:snapped_end]
            chunk_text_stripped = chunk_text.strip()

            if chunk_text_stripped:
                # ── FIX: remove leading punctuation artifacts (e.g. ". Now she runs") ──
                chunk_text_stripped = self._clean_chunk_start(chunk_text_stripped)
                # Advance adjusted_start to match cleaned text in original
                clean_start_offset = chunk_text.find(chunk_text_stripped[0]) if chunk_text_stripped else 0
                adjusted_start_base = chunk_start + (len(chunk_text) - len(chunk_text.lstrip()))

                token_count = self.token_counter.estimate_tokens(chunk_text_stripped)

                # If overlap makes chunk too big, prefer shrinking the END first
                if token_count > self.chunk_tokens:
                    shrink_end = snapped_end
                    while token_count > self.chunk_tokens and shrink_end > chunk_start + 50:
                        shrink_end = self._snap_to_boundary(original_text, shrink_end - 50, direction="backward")
                        chunk_text = original_text[chunk_start:shrink_end]
                        chunk_text_stripped = self._clean_chunk_start(chunk_text.strip())
                        token_count = self.token_counter.estimate_tokens(chunk_text_stripped)
                    snapped_end = shrink_end

                # If STILL too big, move the start forward
                if token_count > self.chunk_tokens:
                    while token_count > self.chunk_tokens and chunk_start < start_pos:
                        chunk_start += 25
                        if overlapped_chunks:
                            prev_meta = overlapped_chunks[-1]
                            chunk_start = min(chunk_start, prev_meta.end_char_index - 1)
                        chunk_text = original_text[chunk_start:snapped_end]
                        chunk_text_stripped = self._clean_chunk_start(chunk_text.strip())
                        token_count = self.token_counter.estimate_tokens(chunk_text_stripped)

                # Recompute final indices after all adjustments
                chunk_text = original_text[chunk_start:snapped_end]
                chunk_text_stripped = self._clean_chunk_start(chunk_text.strip())
                if not chunk_text_stripped:
                    continue

                strip_left = len(chunk_text) - len(chunk_text.lstrip())
                # Account for any additional chars removed by _clean_chunk_start
                extra_clean = len(chunk_text.lstrip()) - len(chunk_text_stripped) if len(chunk_text.lstrip()) > len(chunk_text_stripped) else 0
                strip_right = len(chunk_text) - len(chunk_text.rstrip())

                adjusted_start = chunk_start + strip_left + extra_clean
                adjusted_end = snapped_end - strip_right

                token_count = self.token_counter.estimate_tokens(chunk_text_stripped)

                # Create metadata
                chunk_meta = ChunkMetadata(
                    chunk_id=self._chunk_id_counter,
                    text=chunk_text_stripped,
                    token_count=token_count,
                    character_count=len(chunk_text_stripped),
                    start_char_index=adjusted_start,
                    end_char_index=adjusted_end,
                    source_doc_id=source_doc_id,
                    chunk_index=i,
                    total_chunks=len(chunks_with_positions),
                    title=title,
                    published_date=published_date,
                )

                # Compute overlap from FINAL STORED INDICES
                if overlapped_chunks:
                    prev_meta = overlapped_chunks[-1]

                    overlap_start = max(chunk_meta.start_char_index, prev_meta.start_char_index)
                    overlap_end = min(chunk_meta.end_char_index, prev_meta.end_char_index)

                    if overlap_start < overlap_end:
                        overlap_text = original_text[overlap_start:overlap_end]
                        overlap_tokens = self.token_counter.estimate_tokens(overlap_text)
                    else:
                        overlap_tokens = 0
                        overlap_start = overlap_end = 0

                    chunk_meta.overlap_tokens = overlap_tokens
                    chunk_meta.overlap_start_char_index = overlap_start
                    chunk_meta.overlap_end_char_index = overlap_end

                    if abs(overlap_tokens - self.overlap_tokens) > 10:
                        logger.debug(
                            f"Overlap drift: got {overlap_tokens} tokens "
                            f"(target {self.overlap_tokens}) "
                            f"for chunk_id={chunk_meta.chunk_id} source_doc_id={source_doc_id}"
                        )

                overlapped_chunks.append(chunk_meta)
                self._chunk_id_counter += 1
                prev_end_pos = adjusted_end

        return overlapped_chunks
    
    def _find_chunk_positions(
        self,
        chunks: List[str],
        original_text: str
    ) -> List[Tuple[str, int, int]]:
        """
        Find the position of each chunk in the original text.
        """
        positions = []
        search_start = 0
        
        for chunk in chunks:
            pos = original_text.find(chunk, search_start)
            
            if pos == -1:
                chunk_stripped = chunk.strip()
                if chunk_stripped and chunk_stripped != chunk:
                    pos = original_text.find(chunk_stripped, search_start)
                    if pos != -1:
                        start = pos
                        end = pos + len(chunk_stripped)
                        while start > 0 and original_text[start-1].isspace():
                            start -= 1
                        while end < len(original_text) and original_text[end].isspace():
                            end += 1
                        pos = start
                        chunk = original_text[start:end]
                
                if pos == -1:
                    logger.debug(
                        f"Chunk match failed using fuzzy search: {chunk[:50]}... "
                        f"Using approximate position from search_start={search_start}"
                    )
                    pos = search_start
            
            chunk_end = pos + len(chunk)
            positions.append((chunk, pos, chunk_end))
            search_start = chunk_end
        
        return positions
    
    def reset_counter(self):
        """Reset the chunk ID counter."""
        self._chunk_id_counter = 0


def split_documents_batch(
    documents: List[Dict],
    chunk_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[ChunkMetadata]:
    """
    Batch process multiple documents.
    
    Args:
        documents: List of dicts with 'text', 'id' keys
        chunk_tokens: Target chunk size
        overlap_tokens: Overlap size
        
    Returns:
        List of flattened ChunkMetadata across all documents
        
    Example:
        documents = [
            {'text': 'News article 1...', 'id': 'doc_001'},
            {'text': 'News article 2...', 'id': 'doc_002'},
        ]
        chunks = split_documents_batch(documents)
    """
    splitter = RecursiveCharacterSplitter(
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens
    )
    
    all_chunks = []
    for doc in documents:
        text = doc.get('text', '')
        doc_id = doc.get('id', None)
        chunks = splitter.split(text, source_doc_id=doc_id)
        all_chunks.extend(chunks)
    
    return all_chunks
