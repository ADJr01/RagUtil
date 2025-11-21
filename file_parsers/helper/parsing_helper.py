import os
import re
import unicodedata

# Ligature map used by enterprise NLP pipelines
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

# Pre-compile translation table for ligatures (faster than multiple replace calls)
LIGATURE_TRANS_TABLE = str.maketrans(LIGATURES)

# Pre-compile all regex patterns at module level for maximum performance
# CRITICAL FIX: Exclude font artifacts more carefully to preserve code syntax
FONT_ARTIFACTS_PATTERN = re.compile(
    r"\{(?:sifd|cid\d+|[a-z]{2,4}\d*)\}|"  # Only remove specific font artifacts like {sifd}, {cid1234}
    r"\[(?:sfi\d+|[a-z]{2,4}\d*)\]"  # Only remove specific font artifacts like [sfi2]
)

COMBINED_REMOVAL_PATTERN = re.compile(
    r"[\ue000-\uf8ff]|"  # Private use area (font subset garbage)
    r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]|"  # Control chars (preserve \t=0x09, \n=0x0a, \r=0x0d)
    r"[■◆●◼◾◽▪▫•◊]"  # OCR garbage symbols
)

# FIXED: Include curly braces, square brackets, angle brackets, and common programming chars
INVALID_CHARS_PATTERN = re.compile(r"[^\w\s.,!?:;()'\"/%\-–—\n\{\}\[\]<>=+*&|^~`#@$\\]")

HYPHEN_PATTERN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
BROKEN_LINE_PATTERN = re.compile(r"(?<![\.\?!:\}])\n(?=[a-zA-Z])")  # Don't merge lines after }
HEADER_FOOTER_PATTERN = re.compile(
    r"^\s*(?:Page\s*\d+|\d+|Copyright.*|All rights reserved.*|Chapter\s+\d+.*)\s*$",
    re.MULTILINE | re.IGNORECASE
)
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\n{3,}")


def clean_pdf_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF files.

    Optimized for programming books - preserves code syntax including braces,
    brackets, operators, and other programming symbols.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned and normalized text
    """
    # PASS 1: Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # PASS 2: Replace ligatures using fast translation table
    text = text.translate(LIGATURE_TRANS_TABLE)

    # PASS 3: Remove font artifacts (careful to preserve actual code)
    text = FONT_ARTIFACTS_PATTERN.sub("", text)

    # PASS 4: Combined character removal (single pass over text)
    text = COMBINED_REMOVAL_PATTERN.sub("", text)

    # PASS 5: Remove truly invalid characters (now preserves programming syntax)
    text = INVALID_CHARS_PATTERN.sub("", text)

    # PASS 6: Fix hyphenations: inter-\nnational -> international
    text = HYPHEN_PATTERN.sub("", text)

    # PASS 7: Merge broken lines inside a paragraph (but not after code blocks)
    text = BROKEN_LINE_PATTERN.sub(" ", text)

    # PASS 8: Remove page numbers and repeated headers/footers
    text = HEADER_FOOTER_PATTERN.sub("", text)

    # PASS 9: Normalize multiple spaces
    text = WHITESPACE_PATTERN.sub(" ", text)

    # PASS 10: Collapse multiple newlines
    text = NEWLINE_PATTERN.sub("\n\n", text)

    # PASS 11: Trim individual lines and remove empty lines
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return text.strip()

def is_pdf_file(path):
    return os.path.splitext(path)[1].lower() == ".pdf"
