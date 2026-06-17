"""
rag.py — ISMS-P RAG 검색 + 답변 생성 (Phase 3)

제공 함수:
  retrieve(question, k=4) -> list[dict]   # 관련 기준 검색 (로컬, API 키 불필요)
  answer(question, k=4)   -> dict          # 검색결과를 Claude에 넣어 출처 인용 답변 생성

환각 억제 설계:
  - 검색된 기준만 근거로 답하도록 강제
  - 근거 기준 번호를 [출처: 2.5.1] 형식으로 인용
  - 근거에서 답을 못 찾으면 "찾지 못했습니다"라고 답

CLI:
  python rag.py "위험평가는 어느 기준인가요?"   # 한 질문에 답변
  python rag.py                                   # 샘플 질문 데모 실행
"""

import json
import os
import sys
from functools import lru_cache

# Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    TOP_K,
)

# .env 에서 ANTHROPIC_API_KEY 로드
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
4. 참고 기준에서 질문의 답을 찾을 수 없으면, 추측하지 말고 정확히 "해당 내용은 제공된 ISMS-P 기준에서 찾지 못했습니다." 라고만 답하세요."""


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
            "ANTHROPIC_API_KEY 가 설정되지 않았습니다. .env 파일에 키를 입력하세요."
        )
    return anthropic.Anthropic(api_key=api_key)


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


# ─────────────────────────────────────────────────────────────
# 답변 생성
# ─────────────────────────────────────────────────────────────
def _format_context(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        lines.append(f"({h['id']}) [{h.get('category', '')}] {h.get('title', '')}\n  {h.get('summary', '')}")
    return "\n".join(lines)


def answer(question: str, k: int = TOP_K, hits: list[dict] | None = None) -> dict:
    """질문에 대해 검색 → Claude 답변 생성. {'answer': str, 'sources': list[dict]} 반환.

    hits 를 넘기면 검색을 건너뛰고 재사용한다(UI에서 검색을 먼저 표시한 경우 중복 방지).
    """
    if hits is None:
        hits = retrieve(question, k)
    context = _format_context(hits)
    user_msg = f"[참고 기준]\n{context}\n\n[질문]\n{question}"

    client = _get_anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return {"answer": text.strip(), "sources": hits}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _print_sources(hits: list[dict]) -> None:
    print("[검색된 기준]")
    for h in hits:
        print(f"  - ({h['id']}) {h.get('title', '')}  [{h.get('category', '')}]  유사도={h['similarity']:.3f}")


def _print_answer(question: str) -> None:
    """검색 + Claude 답변 (API 크레딧 필요)."""
    print("\n" + "=" * 64)
    print(f"질문: {question}")
    print("-" * 64)
    result = answer(question)
    print(result["answer"])
    print("-" * 64)
    _print_sources(result["sources"])


def _print_retrieval(question: str) -> None:
    """검색만 (API 호출 없음)."""
    print("\n" + "=" * 64)
    print(f"질문: {question}")
    print("-" * 64)
    _print_sources(retrieve(question))


# 데모/검증용 샘플 질문 (MD 5번 검증셋. 마지막 1개는 '없는 내용' → 찾지 못했습니다 테스트)
SAMPLE_QUESTIONS = [
    "위험평가는 어느 기준에 해당하나요?",
    "접근통제 관련 기준을 알려주세요",
    "개인정보 파기에 대한 요구사항은?",
    "관리체계 수립은 몇 개 기준으로 구성되나요?",
    "랜섬웨어 복구 절차 기준 알려줘",
]


def main() -> None:
    args = sys.argv[1:]

    # -r / --retrieve : 검색만 (API 크레딧 불필요)
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
            try:
                _print_answer(q)
            except Exception as e:
                print(f"[오류] '{q}' 처리 중 예외: {e}")
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
