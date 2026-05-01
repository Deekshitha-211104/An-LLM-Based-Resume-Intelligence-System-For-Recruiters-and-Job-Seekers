import fitz
from docx import Document
from bs4 import BeautifulSoup
import os


def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text


def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()


def extract_text_from_file(file):
    """
    Accepts a Gradio file object or a plain file path string.
    Returns extracted text, or empty string on failure.
    """
    # ── Resolve the file path ──────────────────────────────────────────────────
    if isinstance(file, str):
        file_path = file
    elif hasattr(file, "name"):
        file_path = file.name
    else:
        # Gradio >= 3.x may pass a NamedString or dict
        try:
            file_path = str(file)
        except Exception:
            return ""

    if not file_path or not os.path.exists(file_path):
        print(f"[text_extraction] File not found: {file_path}")
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext == ".docx":
            return extract_text_from_docx(file_path)
        elif ext in (".html", ".htm"):
            return extract_text_from_html(file_path)
        else:
            # Try reading as plain text as a last resort
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"[text_extraction] Error reading {file_path}: {e}")
        return ""