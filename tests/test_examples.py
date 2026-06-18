"""example_situations.py 단위 테스트 — 모든 분야에 예시가 있는지(분야명 매칭) 검증."""
import assess
import example_situations


def test_every_category_has_example():
    cats = [c for _dom, c in assess.categories("ISMS-P")]
    assert len(cats) == 21
    for cat in cats:
        ex = example_situations.SITUATIONS.get(cat)
        assert ex and len(ex) > 80, f"예시 누락/부족: {cat}"


def test_get_returns_generic_fallback():
    assert len(example_situations.get("존재하지않는분야")) > 30
