from pathlib import Path
from src.ingestion.loader import load_pdf
from src.ingestion.cleaner import clean_text
from src.ingestion.chunker import chunk_text

"""
Loop through every PDF in the folder
For each PDF, load its pages
For each page, clean the text
Chunk the cleaned text
For each chunk, store it as a dict with chunk, source, page_number
Collect all chunks into one list and return it
"""

def ingest_documents(folder_path:str) -> list[dict]:
    path = Path(folder_path)
    all_chunks = []

    for pdf_file in path.glob("*.pdf"): #finds all pdf in the folder
        pages = load_pdf(str(pdf_file)) #load the pdf and get its pages as list of dicts with text, page_number, source
        
        for page in pages:
            cleaned = clean_text(page["text"])
            if not cleaned:
                continue    #skip empty pages after cleaning , it is a guard
            chunks = chunk_text(cleaned) #chunk the cleaned text into list of chunks
            
            for chunk in chunks:
                all_chunks.append({
                    "chunk": chunk,
                    "source": page["source"],
                    "page_number": page["page_number"]
                })
    return all_chunks

        