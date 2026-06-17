"""rag.py 단위 테스트. 순수 함수는 모델/API 불필요. 검색 테스트는 인덱스 있을 때만."""
import pytest

import rag
from config import CHROMA_DIR


def test_is_grounded():
    assert rag.is_grounded([{"similarity": 0.9}]) is True
    assert rag.is_grounded([{"similarity": 0.05}]) is False
    assert rag.is_grounded([]) is False


def test_format_context():
    hits = [{"id": "2.5.1", "category": "인증 및 권한관리", "title": "사용자 계정 관리", "summary": "요약문"}]
    ctx = rag._format_context(hits)
    assert "2.5.1" in ctx
    assert "사용자 계정 관리" in ctx
    assert "요약문" in ctx


def test_build_messages():
    hits = [{"id": "1.2.3", "category": "위험 관리", "title": "위험 평가", "summary": "S"}]
    msgs = rag._build_messages("내 질문", hits)
    assert msgs[0]["role"] == "user"
    assert "내 질문" in msgs[0]["content"]
    assert "1.2.3" in msgs[0]["content"]


def test_friendly_error_returns_string():
    msg = rag.friendly_error(ValueError("boom"))
    assert isinstance(msg, str) and msg


def test_answer_gate_blocks_low_similarity(monkeypatch):
    """저유사도 질의는 API 호출 없이 '찾지 못했습니다' (비용 차단 게이트)."""
    monkeypatch.setattr(rag, "retrieve", lambda q, k=4: [{"id": "x", "similarity": 0.1}])
    res = rag.answer("전혀 무관한 질문")
    assert res["grounded"] is False
    assert res["llm_called"] is False
    assert "찾지 못" in res["answer"]


_INDEX_EXISTS = (CHROMA_DIR / "chroma.sqlite3").exists()


@pytest.mark.skipif(not _INDEX_EXISTS, reason="벡터 인덱스 없음 (build_index.py 먼저 실행)")
def test_retrieve_shape():
    hits = rag.retrieve("접근통제", k=4)
    assert len(hits) == 4
    assert all(0.0 <= h["similarity"] <= 1.0 for h in hits)


@pytest.mark.skipif(not _INDEX_EXISTS, reason="벡터 인덱스 없음 (build_index.py 먼저 실행)")
def test_retrieve_relevant_top1():
    # 명확한 질의의 1순위가 기대 기준인지 (검색 품질 스모크)
    top1 = rag.retrieve("개인정보 파기 절차", k=4)[0]["id"]
    assert top1 in {"3.4.1", "3.4.2"}
