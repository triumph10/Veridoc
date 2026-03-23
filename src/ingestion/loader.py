import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(filepath: str) -> list[dict]:
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"No file found at {filepath}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got {path.suffix}")
    
    doc = fitz.open(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page_number": page_num + 1,
                "source": path.name
            })

    doc.close()
    return pages