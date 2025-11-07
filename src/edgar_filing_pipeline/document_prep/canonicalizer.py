from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from typing import List, Optional


SMART_PUNCTUATION_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": ",",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
    "\u200b": "",
}

BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "div",
    "dl",
    "dt",
    "dd",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "ul",
}

LINE_BREAK_TAGS = {"br"}
SKIP_CONTENT_TAGS = {"script", "style", "noscript"}


@dataclass(frozen=True)
class CanonicalCharSource:
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class CanonicalizationResult:
    text: str
    char_sources: List[Optional[CanonicalCharSource]]
    source_id: str


@dataclass(frozen=True)
class _TagToken:
    name: str
    is_start: bool
    is_self_closing: bool


def canonicalize(
    raw_text: str,
    *,
    source_id: str,
    treat_as_html: bool = True,
) -> CanonicalizationResult:
    """Normalize ``raw_text`` while keeping a per-character source map."""
    canonicalizer = _Canonicalizer(
        raw_text=raw_text,
        source_id=source_id,
        treat_as_html=treat_as_html,
    )
    return canonicalizer.run()


class _Canonicalizer:
    """Internal stateful normalizer for producing canonical text."""

    def __init__(self, raw_text: str, source_id: str, treat_as_html: bool) -> None:
        self.raw_text = raw_text
        self.length = len(raw_text)
        self.source_id = source_id
        self.treat_as_html = treat_as_html
        self.convert_newlines_to_space = treat_as_html

        self.canonical_chars: List[str] = []
        self.char_sources: List[Optional[CanonicalCharSource]] = []

        self._pending_block_break = False
        self._last_was_space = False
        self._skip_stack: List[str] = []

    def run(self) -> CanonicalizationResult:
        if self.treat_as_html:
            self._process_html()
        else:
            self._process_plain_text()

        self._trim_trailing_layout()

        canonical_text = "".join(self.canonical_chars)
        return CanonicalizationResult(
            text=canonical_text,
            char_sources=self.char_sources,
            source_id=self.source_id,
        )

    # ------------------------------------------------------------------
    # Core processing loops
    # ------------------------------------------------------------------
    def _process_html(self) -> None:
        i = 0
        while i < self.length:
            if self._skip_stack:
                if self.raw_text[i] == "<":
                    token, new_index = self._parse_tag(i)
                    if token and not token.is_start and token.name == self._skip_stack[-1]:
                        self._skip_stack.pop()
                    i = new_index
                else:
                    i += 1
                continue

            char = self.raw_text[i]
            if char == "<":
                token, new_index = self._parse_tag(i)
                if token is not None:
                    self._handle_tag(token)
                i = new_index
            else:
                next_tag = self.raw_text.find("<", i)
                if next_tag == -1:
                    next_tag = self.length
                self._append_text_segment(i, next_tag)
                i = next_tag

    def _process_plain_text(self) -> None:
        text = self.raw_text.replace("\r\n", "\n").replace("\r", "\n")
        start = 0
        end = len(text)
        self._append_text_segment_plain(text, start, end)

    # ------------------------------------------------------------------
    # Tag parsing and handling
    # ------------------------------------------------------------------
    def _parse_tag(self, index: int) -> tuple[Optional[_TagToken], int]:
        if self.raw_text.startswith("<!--", index):
            end = self.raw_text.find("-->", index + 4)
            if end == -1:
                return None, self.length
            return None, end + 3

        j = index + 1
        while j < self.length and self.raw_text[j].isspace():
            j += 1
        if j >= self.length:
            return None, self.length

        is_start = True
        if self.raw_text[j] == "/":
            is_start = False
            j += 1

        name_start = j
        while j < self.length and (
            self.raw_text[j].isalnum() or self.raw_text[j] in {"-", "_", ":"}
        ):
            j += 1
        name = self.raw_text[name_start:j].lower()

        # Skip attributes and reach end of tag.
        attr_index = j
        quote_char: Optional[str] = None
        while attr_index < self.length and self.raw_text[attr_index] != ">":
            current = self.raw_text[attr_index]
            if current in {"'", '"'}:
                if quote_char is None:
                    quote_char = current
                elif quote_char == current:
                    quote_char = None
            attr_index += 1
        if attr_index >= self.length:
            return None, self.length

        is_self_closing = False
        cursor = attr_index - 1
        while cursor > index and self.raw_text[cursor].isspace():
            cursor -= 1
        if cursor > index and self.raw_text[cursor] == "/":
            is_self_closing = True

        token = _TagToken(name=name, is_start=is_start, is_self_closing=is_self_closing)
        return token, attr_index + 1

    def _handle_tag(self, token: _TagToken) -> None:
        name = token.name

        if token.is_start and name in SKIP_CONTENT_TAGS:
            self._skip_stack.append(name)
            return

        if name in BLOCK_TAGS:
            self._mark_block_break()

        if token.is_start and name in LINE_BREAK_TAGS:
            self._emit_newline(None)
            return

        if not token.is_start and self._skip_stack and name == self._skip_stack[-1]:
            self._skip_stack.pop()

    # ------------------------------------------------------------------
    # Text ingestion helpers
    # ------------------------------------------------------------------
    def _append_text_segment(self, start: int, end: int) -> None:
        pos = start
        while pos < end:
            char = self.raw_text[pos]
            if char == "&":
                entity_end = self.raw_text.find(";", pos + 1, min(end, pos + 32))
                if entity_end != -1:
                    entity = self.raw_text[pos : entity_end + 1]
                    unescaped = unescape(entity)
                    if unescaped != entity:
                        for resolved_char in unescaped:
                            self._emit_character(
                                resolved_char,
                                CanonicalCharSource(self.source_id, pos, entity_end + 1),
                            )
                        pos = entity_end + 1
                        continue
            self._emit_character(
                self.raw_text[pos],
                CanonicalCharSource(self.source_id, pos, pos + 1),
            )
            pos += 1

    def _append_text_segment_plain(self, text: str, start: int, end: int) -> None:
        pos = start
        while pos < end:
            self._emit_character(
                text[pos],
                CanonicalCharSource(self.source_id, pos, pos + 1),
            )
            pos += 1

    # ------------------------------------------------------------------
    # Character emission
    # ------------------------------------------------------------------
    def _emit_character(
        self,
        char: str,
        source: CanonicalCharSource,
    ) -> None:
        if not char:
            return

        normalized = SMART_PUNCTUATION_MAP.get(char, char)

        if normalized in {"\r", "\n"}:
            if self.convert_newlines_to_space:
                self._emit_space(source)
            else:
                self._emit_newline(source)
            return

        if normalized.isspace():
            self._emit_space(source)
            return

        self._apply_pending_paragraph_break()
        self.canonical_chars.append(normalized)
        self.char_sources.append(source)
        self._last_was_space = False

    def _emit_space(self, source: CanonicalCharSource) -> None:
        if self._pending_block_break:
            # Defer until next non-space content.
            return
        if not self.canonical_chars:
            return
        if self.canonical_chars[-1] == "\n":
            return
        if self._last_was_space:
            return
        self.canonical_chars.append(" ")
        self.char_sources.append(source)
        self._last_was_space = True

    def _emit_newline(self, source: Optional[CanonicalCharSource]) -> None:
        self._apply_pending_paragraph_break()
        if self.canonical_chars and self.canonical_chars[-1] == "\n":
            return
        self._strip_trailing_spaces()
        self._append_newline(source)
        self._last_was_space = False

    def _mark_block_break(self) -> None:
        self._pending_block_break = True
        self._last_was_space = False

    def _trim_trailing_layout(self) -> None:
        while self.canonical_chars and self.canonical_chars[-1] in {" ", "\n"}:
            self.canonical_chars.pop()
            self.char_sources.pop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_pending_paragraph_break(self) -> None:
        if not self._pending_block_break:
            return
        self._strip_trailing_spaces()
        if not self.canonical_chars:
            self._pending_block_break = False
            return
        self._append_newline(None)
        if len(self.canonical_chars) < 2 or self.canonical_chars[-2] != "\n":
            self._append_newline(None)
        self._last_was_space = False
        self._pending_block_break = False

    def _append_newline(self, source: Optional[CanonicalCharSource]) -> None:
        self.canonical_chars.append("\n")
        self.char_sources.append(source)

    def _strip_trailing_spaces(self) -> None:
        while self.canonical_chars and self.canonical_chars[-1] == " ":
            self.canonical_chars.pop()
            self.char_sources.pop()
