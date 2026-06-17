"""
eval_retrieval.py — 검색 품질 정량 평가 (Hit@1, Hit@k, MRR)

라벨된 (질의 → 정답 기준 ID 집합) 세트로 임베딩 검색 품질을 측정한다.
"느낌"이 아니라 수치로 검색기를 평가/회귀하기 위함. 인덱스 필요(build_index.py).

실행: python eval_retrieval.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import TOP_K
from rag import retrieve

# (질의, 정답으로 인정할 기준 ID 집합) — 도메인 전반을 커버하도록 구성
EVAL_SET = [
    ("위험평가는 어느 기준인가요?", {"1.2.3"}),
    ("정보자산 식별 및 분류", {"1.2.1"}),
    ("경영진 참여 보고체계", {"1.1.1"}),
    ("접근통제 관련 기준", {"2.6.1", "2.6.2", "2.6.3", "2.6.4", "2.6.5", "2.6.6", "2.6.7"}),
    ("비밀번호 관리 규칙", {"2.5.4"}),
    ("사용자 계정 등록과 권한 부여", {"2.5.1"}),
    ("데이터 암호화 적용", {"2.7.1", "2.7.2"}),
    ("접속기록 로그 보관 점검", {"2.9.4", "2.9.5"}),
    ("백업 및 복구 절차", {"2.9.3"}),
    ("악성코드 랜섬웨어 대응", {"2.10.9"}),
    ("취약점 점검 조치", {"2.11.2"}),
    ("재해 복구 RTO 목표시간", {"2.12.1", "2.12.2"}),
    ("개인정보 수집 동의", {"3.1.1"}),
    ("CCTV 영상정보처리기기 설치", {"3.1.6"}),
    ("개인정보 파기 절차", {"3.4.1", "3.4.2"}),
    ("개인정보 국외 이전", {"3.3.4"}),
    ("개인정보 처리업무 위탁 수탁자", {"3.3.2"}),
    ("정보주체 열람 정정 권리", {"3.5.2"}),
    ("교육 및 훈련 계획", {"2.2.4"}),
    ("망분리 인터넷 접속 통제", {"2.6.7"}),
]


def evaluate(k: int = TOP_K) -> None:
    hit1 = hitk = 0
    rr_sum = 0.0
    rows = []
    for q, gold in EVAL_SET:
        ids = [h["id"] for h in retrieve(q, k=k)]
        is_hit1 = ids and ids[0] in gold
        rank = next((i for i, cid in enumerate(ids, 1) if cid in gold), 0)
        hit1 += int(bool(is_hit1))
        hitk += int(rank > 0)
        rr_sum += (1.0 / rank) if rank else 0.0
        mark = "OK " if rank == 1 else ("hit" if rank else "MISS")
        rows.append((mark, rank or "-", ids[0] if ids else "-", q, gold))

    n = len(EVAL_SET)
    print("=" * 70)
    print(f"  검색 품질 평가 — 평가셋 {n}개 · top-k={k}")
    print("=" * 70)
    print(f"  Hit@1  : {hit1}/{n} = {hit1 / n:.1%}   (1순위가 정답)")
    print(f"  Hit@{k} : {hitk}/{n} = {hitk / n:.1%}   (top-{k} 안에 정답)")
    print(f"  MRR    : {rr_sum / n:.3f}            (정답의 평균 역순위)")
    print("-" * 70)
    for mark, rank, top1, q, gold in rows:
        print(f"  [{mark}] rank={rank:<2} 1순위={top1:<7} ← {q}  (정답 {sorted(gold)})")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
