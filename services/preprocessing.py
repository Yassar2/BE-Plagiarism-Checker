# services/preprocessing.py

import re
import io
import PyPDF2
from docx import Document

word_re = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_text(text):
    return [t.lower() for t in word_re.findall(str(text))]


def jaccard_similarity(a, b):
    set_a = set(tokenize_text(a))
    set_b = set(tokenize_text(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_stream):
    doc = Document(file_stream)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_file(file):
    filename = file.filename.lower()
    file_stream = io.BytesIO(file.read())

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_stream)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_stream)
    elif filename.endswith(".txt"):
        file_stream.seek(0)
        return file_stream.read().decode("utf-8")
    else:
        raise ValueError("Format file tidak didukung (PDF, DOCX, TXT).")
