"""
rag.py — ISMS-P RAG 검색 + 답변 생성

제공 함수:
  retrieve(question, k=4)      -> list[dict]   # 관련 기준 검색 (로컬, API 키 불필요)
  is_grounded(hits)            -> bool         # 최상위 유사도가 임계값 이상인지
  answer(question, k=4)        -> dict          # 비스트리밍 답변 (CLI/테스트용)
  stream_answer(question, k=4) -> Iterator[str] # 스트리밍 답변 (Streamlit용)
  friendly_error(exc)          -> str          # 예외를 사용자용 한국어 메시지로

환각 억제 설계:
  - 검색된 기준만 근거로 답하도록 강제
  - 근거 기준 번호를 [출처: 2.5.1] 형식으로 인용
  - 검색 최상위 유사도가 임계값 미만이면 Claude를 호출하지 않고 "찾지 못했습니다"
    (오프토픽 질의의 API 비용 차단 + 즉시 응답)

CLI:
  python rag.py "위험평가는 어느 기준인가요?"   # 답변
  python rag.py -r "개인정보 파기"               # 검색만 (API 불필요)
  python rag.py                                   # 샘플 질문 데모
"""

import json
import os
import sys
from functools import lru_cache
from typing import Iterator

# Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sqlite_fix  # noqa: F401 — chromadb import 전에 sqlite 교체
import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CLAUDE_MODEL,
    COLLECTION_NAME,
    DATA_PATH,
    EMBED_MODEL_NAME,
    MAX_TOKENS,
    NOT_FOUND_MESSAGE,
    REQUEST_TIMEOUT,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# .env 에서 ANTHROPIC_API_KEY 로드 (배포 환경은 app.py가 Secrets를 주입)
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 환각 억제 프롬프트
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 ISMS-P(정보보호 및 개인정보보호 관리체계) 인증기준을 안내하는 도우미입니다.
오직 사용자 메시지의 '참고 기준'에 제시된 내용만을 근거로 답하세요.

반드시 지킬 규칙:
1. 참고 기준에 있는 내용만 사용하세요. 참고 기준에 없는 사실은 절대 지어내지 마세요.
2. 어려운 용어는 풀어서, 쉽고 명확한 한국어로 설명하세요.
3. 답변에 근거로 사용한 기준 번호를 [출처: 2.5.1] 형식으로 표기하세요. 여러 개이면 [출처: 1.2.3, 2.6.1] 과 같이 적으세요.
4. 참고 기준에서 질문의 답을 찾을 수 없으면, 추측하지 말고 정확히 "해당 내용은 제공된 ISMS-P 기준에서 찾지 못했습니다." 라고만 답하세요.
5. 참고 기준의 요약문은 공식 원문이 아니라 비공식 요약임을 감안하여, 단정적 표현보다 안내하는 어조를 사용하세요."""


# ─────────────────────────────────────────────────────────────
# 리소스 로더 (한 번만 로드하여 재사용)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"벡터스토어 '{COLLECTION_NAME}' 를 찾을 수 없습니다. "
            f"먼저 'python build_index.py' 를 실행하세요. (원인: {e})"
        )


@lru_cache(maxsize=1)
def _get_criteria_map() -> dict:
    """id -> 기준(dict) 매핑. 검색 결과에 summary 등 원문 필드를 붙이기 위함."""
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"]: c for c in data}


@lru_cache(maxsize=1)
def _get_anthropic() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY 가 설정되지 않았습니다. .env 또는 Secrets를 확인하세요."
        )
    # 타임아웃을 명시해 API 지연 시 앱이 무한정 멈추지 않도록 함
    return anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT)


# ─────────────────────────────────────────────────────────────
# 검색
# ─────────────────────────────────────────────────────────────
def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """질문과 가장 관련 있는 ISMS-P 기준 k개를 반환한다 (유사도 내림차순)."""
    model = _get_model()
    collection = _get_collection()
    cmap = _get_criteria_map()

    q_emb = model.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)

    hits = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        crit = dict(cmap.get(cid, {"id": cid}))
        crit["similarity"] = round(1 - dist, 4)  # 코사인 유사도
        hits.append(crit)
    return hits


def is_grounded(hits: list[dict]) -> bool:
    """최상위 결과의 유사도가 임계값 이상이면 True (= LLM 호출 가치 있음)."""
    return bool(hits) and hits[0].get("similarity", 0.0) >= SIMILARITY_THRESHOLD


# ─────────────────────────────────────────────────────────────
# 답변 생성
# ─────────────────────────────────────────────────────────────
def _format_context(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        lines.append(
            f"({h['id']}) [{h.get('category', '')}] {h.get('title', '')}\n  {h.get('summary', '')}"
        )
    return "\n".join(lines)


def _build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = _format_context(hits)
    user_msg = f"[참고 기준]\n{context}\n\n[질문]\n{question}"
    return [{"role": "user", "content": user_msg}]


def answer(question: str, k: int = TOP_K, hits: list[dict] | None = None) -> dict:
    """비스트리밍 답변. {'answer', 'sources', 'grounded', 'llm_called'} 반환.

    검색 최상위 유사도가 임계값 미만이면 Claude를 호출하지 않고 '찾지 못했습니다'.
    API 예외는 호출자에게 전파된다(friendly_error로 변환 가능).
    """
    if hits is None:
        hits = retrieve(question, k)

    if not is_grounded(hits):
        return {"answer": NOT_FOUND_MESSAGE, "sources": hits, "grounded": False, "llm_called": False}

    client = _get_anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=_build_messages(question, hits),
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return {"answer": text.strip(), "sources": hits, "grounded": True, "llm_called": True}


def stream_answer(question: str, k: int = TOP_K, hits: list[dict] | None = None) -> Iterator[str]:
    """스트리밍 답변 제너레이터 (Streamlit st.write_stream 용). 텍스트 청크를 yield.

    근거 부족이면 즉시 '찾지 못했습니다'를 yield하고 종료(API 호출 없음).
    API 예외는 이터레이션 중 전파되므로 호출자가 friendly_error로 처리한다.
    """
    if hits is None:
        hits = retrieve(question, k)

    if not is_grounded(hits):
        yield NOT_FOUND_MESSAGE
        return

    client = _get_anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=_build_messages(question, hits),
    ) as stream:
        for text in stream.text_stream:
            yield text


def friendly_error(exc: Exception) -> str:
    """API 예외를 사용자용 한국어 메시지로. 구체 타입부터 검사(문자열 매칭 최소화)."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "API 키 인증에 실패했습니다. ANTHROPIC_API_KEY(.env 또는 Secrets)를 확인하세요."
    if isinstance(exc, anthropic.RateLimitError):
        return "요청이 많아 잠시 제한되었습니다. 잠시 후 다시 시도하세요."
    if isinstance(exc, anthropic.APITimeoutError):
        return "응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
    if isinstance(exc, anthropic.APIConnectionError):
        return "네트워크 연결 문제로 응답을 받지 못했습니다."
    if isinstance(exc, anthropic.BadRequestError):
        # 크레딧 부족만 메시지 기반 판별 (전용 예외 타입이 없음)
        msg = str(getattr(exc, "message", "") or exc).lower()
        if "credit balance" in msg:
            return "Anthropic API 크레딧 잔액이 부족합니다. console.anthropic.com에서 충전 후 다시 시도하세요."
        return "요청이 거부되었습니다(잘못된 요청)."
    if isinstance(exc, anthropic.APIError):
        return "Claude API 오류가 발생했습니다. 잠시 후 다시 시도하세요."
    return "답변 생성 중 오류가 발생했습니다."


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _print_sources(hits: list[dict]) -> None:
    print("[검색된 기준]")
    for h in hits:
        print(f"  - ({h['id']}) {h.get('title', '')}  [{h.get('category', '')}]  유사도={h['similarity']:.3f}")


def _print_answer(question: str) -> None:
    print("\n" + "=" * 64)
    print(f"질문: {question}")
    print("-" * 64)
    try:
        result = answer(question)
        print(result["answer"])
        if not result["llm_called"]:
            print("  (근거 유사도가 낮아 Claude 호출 없이 응답)")
    except Exception as e:  # noqa: BLE001 - CLI에서는 친절히 변환
        print(f"[오류] {friendly_error(e)}")
        result = {"sources": retrieve(question)}
    print("-" * 64)
    _print_sources(result["sources"])


def _print_retrieval(question: str) -> None:
    print("\n" + "=" * 64)
    print(f"질문: {question}")
    print("-" * 64)
    _print_sources(retrieve(question))


SAMPLE_QUESTIONS = [
    "위험평가는 어느 기준에 해당하나요?",
    "접근통제 관련 기준을 알려주세요",
    "개인정보 파기에 대한 요구사항은?",
    "관리체계 수립은 몇 개 기준으로 구성되나요?",
    "랜섬웨어 복구 절차 기준 알려줘",
]


def main() -> None:
    args = sys.argv[1:]
    retrieve_only = bool(args) and args[0] in ("-r", "--retrieve")
    if retrieve_only:
        args = args[1:]

    if retrieve_only:
        questions = [" ".join(args)] if args else SAMPLE_QUESTIONS
        print("검색(retrieve)만 실행합니다 — Claude API 호출 없음.")
        for q in questions:
            _print_retrieval(q)
    elif args:
        _print_answer(" ".join(args))
    else:
        print("샘플 질문 데모(검색+답변)를 실행합니다. (검색만: python rag.py -r)")
        for q in SAMPLE_QUESTIONS:
            _print_answer(q)
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
