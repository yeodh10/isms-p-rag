"""
docparse.py — 업로드 문서(PDF/DOCX/TXT)에서 텍스트 추출.

사전 자가점검에서 회사 현황·증적 문서를 올리면 텍스트를 뽑아 입력창을 채우는 데 쓴다.
"""

import io


def extract_text(name: str, data: bytes) -> str:
    """파일명 확장자로 형식을 판별해 텍스트를 추출한다."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        return _from_pdf(data)
    if lower.endswith(".docx"):
        return _from_docx(data)
    if lower.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"지원하지 않는 형식입니다: {name} (PDF/DOCX/TXT만 지원)")


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _from_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()
