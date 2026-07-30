from pathlib import Path
import os
import re

# Docling imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import BaseChunker

# Embedding & Vector Database imports
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Setup Directories
# ---------------------------------------------------------------------------
html_dir = Path("tmp/html")
docling_dir = Path("tmp/docling")
chroma_dir = Path("tmp/chromadb")

html_dir.mkdir(parents=True, exist_ok=True)
docling_dir.mkdir(parents=True, exist_ok=True)
chroma_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parse HTML File with Docling & Save Outputs
# ---------------------------------------------------------------------------
input_html_path = "Legea nr.227_2015.html"  # Update this with your actual file path

converter = DocumentConverter()
result = converter.convert(input_html_path)
docling_doc = result.document

# Save as HTML (visually check parsing stage)
output_html_path = html_dir / f"{Path(input_html_path).stem}.html"
with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(docling_doc.export_to_html())

# Save as JSON (reuse structured representation later)
output_json_path = docling_dir / f"{Path(input_html_path).stem}.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    f.write(docling_doc.model_dump_json())

print(f" Saved HTML check to: {output_html_path}")
print(f" Saved JSON to: {output_json_path}")

# ---------------------------------------------------------------------------
# 2 & 3. Chunking & Metadata Extraction
# ---------------------------------------------------------------------------
chunker = HybridChunker()
chunks_iter = chunker.chunk(docling_doc)

documents = []
metadatas = []
ids = []

# Regex patterns for matching structural elements (e.g., Title, Chapter, Article)
regex_title = re.compile(r"Titlul\s+([IVXLCDM\d]+)[\s:.-]*(.*)", re.IGNORECASE)
regex_chapter = re.compile(r"Capitolul\s+([IVXLCDM\d]+)[\s:.-]*(.*)", re.IGNORECASE)
regex_article = re.compile(r"Articolul\s+(\d+)|Art\.\s*(\d+)", re.IGNORECASE)

for idx, chunk in enumerate(chunks_iter):
    chunk_text = chunk.text
    
    # Extract metadata context directly from Docling's structural headings
    headings = getattr(chunk.meta, "headings", [])
    
    metadata = {
        "title": "",
        "title_no": 0,
        "chapter": "",
        "chapter_no": 0,
        "article": "",
        "article_no": 0
    }
    
    # Process headings to populate structural metadata
    for heading in headings:
        # Match Title
        match_t = regex_title.search(heading)
        if match_t:
            metadata["title_no"] = match_t.group(1)
            metadata["title"] = match_t.group(2).strip() or heading
            
        # Match Chapter
        match_c = regex_chapter.search(heading)
        if match_c:
            metadata["chapter_no"] = match_c.group(1)
            metadata["chapter"] = match_c.group(2).strip() or heading

    # Match Article from text body or headings
    match_a = regex_article.search(chunk_text)
    if match_a:
        metadata["article_no"] = int(match_a.group(1) or match_a.group(2))
        metadata["article"] = f"Articolul {metadata['article_no']}"

    documents.append(chunk_text)
    metadatas.append(metadata)
    ids.append(f"doc_chunk_{idx}")

print(f" Extracted {len(documents)} chunks with metadata.")

# ---------------------------------------------------------------------------
# 4. Generate Embeddings (sentence-transformers/all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()

# ---------------------------------------------------------------------------
# 5. Save to ChromaDB (SQLite Persisted Store)
# ---------------------------------------------------------------------------
client = chromadb.PersistentClient(path=str(chroma_dir))
collection = client.get_or_create_collection(name="legal_docling_chunks")

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f" Successfully inserted {len(documents)} items into ChromaDB at '{chroma_dir}'.")
