"""
build_index.py — ISMS-P 인증기준 임베딩 → Chroma 벡터스토어 구축 (Phase 2)

동작:
  1. data/isms_criteria.json 로드
  2. 한국어 임베딩 모델 로드 (최초 1회 다운로드)
  3. 각 기준을 "[분야] 제목 + 요약" 텍스트로 임베딩
  4. Chroma 영구 컬렉션에 메타데이터(id/domain/category/title)와 함께 저장
  5. 샘플 검색("접근통제")으로 정상 동작 확인

사용법:
  python build_index.py          # 인덱스 (재)생성
한 번 실행하면 data/chroma/ 에 저장되어 rag.py / app.py 가 재사용한다.
"""

import json
import sys

# Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sqlite_fix  # noqa: F401 — chromadb import 전에 sqlite 교체
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_PATH,
    EMBED_MODEL_NAME,
    TOP_K,
)


def make_document(criterion: dict) -> str:
    """임베딩 대상 텍스트. 분야(category)를 포함시켜 '접근통제' 같은 분야명 질의에도 잘 매칭되게 한다."""
    return f"[{criterion['category']}] {criterion['title']}\n{criterion['summary']}"


def load_criteria() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_index() -> None:
    print("=" * 60)
    print("  ISMS-P 벡터 인덱스 구축")
    print("=" * 60)

    # 1) 데이터 로드
    criteria = load_criteria()
    print(f"[1/5] 기준 데이터 로드: {len(criteria)}개")

    # 2) 임베딩 모델 로드 (최초 실행 시 다운로드)
    print(f"[2/5] 임베딩 모델 로드 중: {EMBED_MODEL_NAME}")
    print("       (최초 1회는 다운로드로 수 분 걸릴 수 있습니다)")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    # sentence-transformers 버전에 따라 메서드명이 다름 (신: get_embedding_dimension)
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"       모델 로드 완료 (임베딩 차원: {get_dim()})")

    # 3) Chroma 영구 클라이언트 + 컬렉션 (기존 것이 있으면 지우고 재생성)
    print(f"[3/5] Chroma 컬렉션 준비: {COLLECTION_NAME}  ->  {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("       기존 컬렉션 삭제 후 재생성")
    except Exception:
        print("       신규 컬렉션 생성")
    # 코사인 거리 사용 (정규화된 임베딩과 함께)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4) 임베딩 후 저장
    print(f"[4/5] {len(criteria)}개 기준 임베딩 및 저장 중...")
    documents = [make_document(c) for c in criteria]
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.add(
        ids=[c["id"] for c in criteria],
        embeddings=embeddings,
        documents=documents,
        metadatas=[
            {
                "id": c["id"],
                "domain": c["domain"],
                "category": c["category"],
                "title": c["title"],
            }
            for c in criteria
        ],
    )
    stored = collection.count()
    print(f"       저장 완료: {stored}개")
    if stored != len(criteria):
        print(f"       [경고] 저장 개수({stored})가 기준 개수({len(criteria)})와 다릅니다.")

    # 5) 샘플 검색 — 완료 기준 확인
    print("[5/5] 샘플 검색 테스트: '접근통제'")
    sample_search(collection, model, "접근통제")

    print("=" * 60)
    print("  인덱스 구축 완료 — data/chroma/ 에 저장됨")
    print("=" * 60)


def sample_search(collection, model, query: str, k: int = TOP_K) -> None:
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    print(f"       질의: \"{query}\"  ->  상위 {k}개")
    for rank, (cid, meta, dist) in enumerate(zip(ids, metas, dists), start=1):
        sim = 1 - dist  # 코사인 유사도
        print(f"        {rank}. [{cid}] {meta['title']}  ({meta['category']})  유사도={sim:.3f}")


def index_exists() -> bool:
    """벡터 인덱스(컬렉션)가 존재하고 비어있지 않은지 확인."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(COLLECTION_NAME).count() > 0
    except Exception:
        return False


def ensure_index() -> None:
    """인덱스가 없으면 빌드한다(배포 환경 첫 실행 대비). 이미 있으면 아무것도 하지 않는다."""
    if not index_exists():
        build_index()


if __name__ == "__main__":
    build_index()
