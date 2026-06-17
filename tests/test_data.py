"""데이터 무결성 테스트 — 모델·API 불필요(빠름). 공식 KISA 2023.11 기준 검증."""
import json
from collections import Counter
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "isms_criteria.json"
REQUIRED = ["id", "domain", "category", "title", "summary"]
EXPECTED_BY_DOMAIN = {
    "관리체계 수립 및 운영": 16,
    "보호대책 요구사항": 64,
    "개인정보 처리단계별 요구사항": 21,
}


@pytest.fixture(scope="module")
def criteria():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_total_count(criteria):
    assert len(criteria) == 101


def test_domain_distribution(criteria):
    assert dict(Counter(c["domain"] for c in criteria)) == EXPECTED_BY_DOMAIN


def test_unique_ids(criteria):
    ids = [c["id"] for c in criteria]
    assert len(ids) == len(set(ids))


def test_id_format(criteria):
    for c in criteria:
        parts = c["id"].split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), c["id"]


def test_required_fields_nonempty(criteria):
    for c in criteria:
        for field in REQUIRED:
            value = c.get(field)
            assert isinstance(value, str) and value.strip(), f"{c.get('id')}:{field}"


def test_category_counts(criteria):
    by_cat = Counter(c["category"] for c in criteria)
    assert by_cat["접근통제"] == 7
    assert by_cat["인증 및 권한관리"] == 6
    assert by_cat["재해 복구"] == 2  # 띄어쓰기 수정(재해복구→재해 복구) 회귀 방지


def test_official_titles(criteria):
    """공식 KISA 2023.11 안내서 대조로 확정한 명칭 회귀 방지."""
    by_id = {c["id"]: c for c in criteria}
    assert by_id["3.3.4"]["title"] == "개인정보 국외이전"        # '의' 제거 확정
    assert by_id["3.1.7"]["title"] == "마케팅 목적의 개인정보 수집·이용"  # 위키 라벨 아님
    assert by_id["1.2.3"]["title"] == "위험 평가"
    assert by_id["2.6.1"]["title"] == "네트워크 접근"
