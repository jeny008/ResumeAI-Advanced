from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from docx import Document
class ParseError(Exception): pass
def parse_resume(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(content)).pages)
        elif ext == ".docx":
            d = Document(BytesIO(content))
            text = "\n".join(p.text for p in d.paragraphs)
            for table in d.tables:
                for row in table.rows: text += "\n" + " | ".join(c.text for c in row.cells)
        elif ext == ".txt": text = content.decode("utf-8", errors="replace")
        else: raise ParseError("Supported formats: PDF, DOCX and TXT.")
    except ParseError: raise
    except Exception as e: raise ParseError(f"Could not parse file: {e}") from e
    text = "\n".join(x.strip() for x in text.splitlines() if x.strip())
    if len(text) < 40: raise ParseError("Not enough text extracted. Scanned PDFs need OCR; try a text-based PDF.")
    return text
