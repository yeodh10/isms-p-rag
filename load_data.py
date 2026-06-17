"""
load_data.py — ISMS-P 인증기준 데이터 로드 및 검증 스크립트 (Phase 1)

역할:
  - data/isms_criteria.json 을 로드한다.
  - 전체 개수(101개) 및 영역별 개수(16 / 64 / 21)를 검증한다.
  - 필수 필드 누락/공백, ID 중복, ID 형식을 점검하여 리포트로 출력한다.

사용법:
  python load_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Windows 콘솔(cp949)에서 한글/유니코드 출력이 깨지거나 크래시 나지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 데이터 파일 경로 (이 스크립트 위치 기준)
DATA_PATH = Path(__file__).parent / "data" / "isms_criteria.json"

# 검증 기준값
EXPECTED_TOTAL = 101
EXPECTED_BY_DOMAIN = {
    "관리체계 수립 및 운영": 16,
    "보호대책 요구사항": 64,
    "개인정보 처리단계별 요구사항": 21,
}
# 비어 있으면 안 되는 필수 필드
REQUIRED_FIELDS = ["id", "domain", "category", "title", "summary"]
# 있으면 좋지만 없어도 통과하는 선택 필드
OPTIONAL_FIELDS = ["checklist"]


def load_criteria(path: Path) -> list[dict]:
    """JSON 파일을 읽어 기준 리스트를 반환한다."""
    if not path.exists():
        print(f"[오류] 데이터 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 실패: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("[오류] 최상위 구조는 리스트(JSON 배열)여야 합니다.")
        sys.exit(1)
    return data


def validate(criteria: list[dict]) -> bool:
    """기준 데이터를 검증하고 리포트를 출력한다. 모든 검사를 통과하면 True."""
    ok = True
    print("=" * 60)
    print("  ISMS-P 인증기준 데이터 검증 리포트")
    print("=" * 60)

    # 1) 전체 개수
    total = len(criteria)
    mark = "OK" if total == EXPECTED_TOTAL else "FAIL"
    if total != EXPECTED_TOTAL:
        ok = False
    print(f"[{mark}] 전체 기준 개수: {total} (기대값 {EXPECTED_TOTAL})")

    # 2) 영역별 개수
    print("-" * 60)
    print("영역별 개수:")
    by_domain = Counter(item.get("domain", "(미지정)") for item in criteria)
    for domain, expected in EXPECTED_BY_DOMAIN.items():
        actual = by_domain.get(domain, 0)
        mark = "OK" if actual == expected else "FAIL"
        if actual != expected:
            ok = False
        print(f"  [{mark}] {domain}: {actual} (기대값 {expected})")
    # 기대 목록에 없는 영역이 섞여 있는지 확인
    unknown = set(by_domain) - set(EXPECTED_BY_DOMAIN)
    if unknown:
        ok = False
        print(f"  [FAIL] 알 수 없는 영역명: {sorted(unknown)}")

    # 3) ID 중복 / 형식
    print("-" * 60)
    ids = [item.get("id", "") for item in criteria]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        ok = False
        print(f"[FAIL] 중복된 ID: {sorted(dup)}")
    else:
        print("[OK] ID 중복 없음")

    # ID 형식 검사 (예: '2.5.1' — 점으로 구분된 3개 숫자)
    bad_id_format = [
        i for i in ids
        if not (i.count(".") == 2 and all(p.isdigit() for p in i.split(".")))
    ]
    if bad_id_format:
        ok = False
        print(f"[FAIL] ID 형식 이상 (x.y.z 형태 아님): {bad_id_format}")
    else:
        print("[OK] 모든 ID가 x.y.z 형식")

    # 4) 필수 필드 누락/공백 리포트
    print("-" * 60)
    print("필수 필드 누락/공백 점검:")
    field_problems = []  # (id, field) 목록
    for item in criteria:
        cid = item.get("id", "(ID없음)")
        for field in REQUIRED_FIELDS:
            value = item.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                field_problems.append((cid, field))

    if field_problems:
        ok = False
        for cid, field in field_problems:
            print(f"  [FAIL] {cid}: '{field}' 필드 누락 또는 공백")
    else:
        print("  [OK] 모든 기준이 필수 필드를 채우고 있음")

    # 5) 선택 필드(checklist) 채움 현황 — 경고만, 통과 여부에는 영향 없음
    print("-" * 60)
    for field in OPTIONAL_FIELDS:
        filled = sum(
            1 for item in criteria
            if isinstance(item.get(field), str) and item.get(field).strip() != ""
        )
        print(f"[정보] 선택 필드 '{field}' 채움: {filled}/{total}")

    # 최종 결과
    print("=" * 60)
    print(f"  최종 결과: {'PASS — Phase 1 완료 기준 충족' if ok else 'FAIL — 위 항목 수정 필요'}")
    print("=" * 60)
    return ok


def main() -> None:
    criteria = load_criteria(DATA_PATH)
    ok = validate(criteria)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
