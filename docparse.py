"""
docparse.py — 업로드 문서(PDF/DOCX/TXT)에서 텍스트 추출.

사전 자가점검에서 회사 현황·증적 문서를 올리면 텍스트를 뽑아 입력창을 채우는 데 쓴다.

공개 앱이라 업로드 기반 DoS(거대 파일·다페이지·zip-bomb)를 막기 위해 상한을 둔다:
파일 크기 10MB · PDF 50페이지 · 추출 텍스트 20만자(초과분 절단).
"""

import io

MAX_BYTES = 10 * 1024 * 1024  # 업로드 파일 크기 상한 (10MB)
MAX_PDF_PAGES = 50            # PDF에서 추출할 최대 페이지 수
MAX_CHARS = 200_000           # 추출 텍스트 길이 상한(메모리·다운스트림 보호)


def extract_text(name: str, data: bytes) -> str:
    """파일명 확장자로 형식을 판별해 텍스트를 추출한다(상한 적용)."""
    if data is None or len(data) == 0:
        return ""
    if len(data) > MAX_BYTES:
        mb = MAX_BYTES // (1024 * 1024)
        raise ValueError(f"파일이 너무 큽니다({len(data) // (1024 * 1024)}MB). 최대 {mb}MB까지 지원합니다.")

    lower = name.lower()
    if lower.endswith(".pdf"):
        text = _from_pdf(data)
    elif lower.endswith(".docx"):
        text = _from_docx(data)
    elif lower.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"지원하지 않는 형식입니다: {name} (PDF/DOCX/TXT만 지원)")

    return text[:MAX_CHARS]


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:MAX_PDF_PAGES]  # 페이지 수 상한(다페이지 DoS 방어)
    return "\n".join((page.extract_text() or "") for page in pages).strip()


def _from_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    total = 0
    for p in document.paragraphs:  # MAX_CHARS 도달 시 조기 중단(zip-bomb류 방어)
        parts.append(p.text)
        total += len(p.text) + 1
        if total > MAX_CHARS:
            break
    return "\n".join(parts).strip()
