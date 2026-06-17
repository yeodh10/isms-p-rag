"""assess.py 단위 테스트 — 순수 함수(모델/API 불필요)."""
import assess


def test_categories_isms_p_count():
    # 분야 수: 1.1~1.4(4) + 2.1~2.12(12) + 3.1~3.5(5) = 21
    assert len(assess.categories("ISMS-P")) == 21


def test_categories_isms_excludes_privacy():
    cats = assess.categories("ISMS")
    assert len(cats) == 16  # 1장(4) + 2장(12)
    assert all(dom != "개인정보 처리단계별 요구사항" for dom, _cat in cats)


def test_criteria_in_category():
    crit = assess.criteria_in_category("접근통제")
    assert len(crit) == 7
    assert all(c["id"].startswith("2.6") for c in crit)


def test_strip_json_plain():
    assert assess._strip_json('{"a": 1}') == '{"a": 1}'


def test_strip_json_fenced():
    assert assess._strip_json('```json\n{"a": 1}\n```').strip() == '{"a": 1}'
