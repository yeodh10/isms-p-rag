"""
assess.py — ISMS-P 사전 자가점검 (AI 보조 갭분석)

분야(category)를 고르고 회사의 '현황/증적'을 입력하면, 그 분야의 각 인증기준에 대해
충족/부분충족/미흡/해당없음을 LLM이 보조 판단하고 보완점을 제시한다.

⚠️ AI 보조 의견일 뿐 공식 결함 판단이 아니다. 실제 인증 가부는 KISA 심사원/인증기관이 결정.

제공 함수:
  categories(scope)            -> list[(domain, category)]   # 점검 분야 목록
  criteria_in_category(cat)    -> list[dict]                  # 분야 내 기준
  assess_category(cat, 현황)   -> list[dict]                  # 기준별 보조 판단
"""

import json
from functools import lru_cache

from config import ASSESS_MAX_TOKENS, CLAUDE_MODEL, DATA_PATH
from rag import _get_anthropic  # 클라이언트(타임아웃 포함) 재사용

VALID_STATUS = {"충족", "부분충족", "미흡", "해당없음"}

ASSESS_SYSTEM = """당신은 ISMS-P 인증 사전 자가점검을 돕는 도우미입니다.
회사가 제출한 '현황/증적'을 주어진 '인증기준'과 비교해, 각 기준별 충족 여부를 보조 판단합니다.

규칙:
1. 오직 제공된 인증기준과 회사 현황만 근거로 판단하세요. 현황에 없는 내용을 있다고 가정하지 마세요.
2. status는 다음 중 하나입니다:
   - "충족": 현황이 요구사항을 명확히 만족
   - "부분충족": 일부만 만족하거나 근거가 불명확
   - "미흡": 만족하지 못하거나 관련 근거가 현황에 없음
   - "해당없음": 조직 특성상 적용 대상이 아님
3. rationale: 현황의 어느 부분을 근거로 판단했는지 한국어로 간결히. 관련 정보가 없으면 "제출된 현황에 관련 정보 없음"이라고 쓰세요.
4. recommendation: 부분충족·미흡일 때 구체적인 보완 권고를 적고, 충족·해당없음이면 빈 문자열("")로 두세요.
5. 당신의 판단은 보조 의견이며 공식 결함 판단이 아닙니다.

출력은 아래 JSON 객체 하나만 출력하세요(코드펜스·설명 금지). 제시된 모든 기준을 제시된 순서대로 포함하세요.
{"assessments":[{"id":"기준ID","status":"충족|부분충족|미흡|해당없음","rationale":"...","recommendation":"..."}]}"""


@lru_cache(maxsize=1)
def _criteria() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def categories(scope: str = "ISMS-P") -> list[tuple]:
    """점검 가능한 (domain, category) 목록을 정의 순서대로 반환. scope='ISMS'면 개인정보(3장) 제외."""
    crit = _criteria()
    if scope == "ISMS":
        crit = [c for c in crit if c["id"].split(".")[0] in {"1", "2"}]
    seen = []
    for c in crit:
        key = (c["domain"], c["category"])
        if key not in seen:
            seen.append(key)
    return seen


def criteria_in_category(category: str) -> list[dict]:
    return [c for c in _criteria() if c["category"] == category]


def _strip_json(text: str) -> str:
    """코드펜스 등이 섞여도 JSON 본문만 추출."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end != -1 else t


# 구조화 출력 스키마 — 모델이 항상 유효한 JSON을 내도록 강제(파싱 안정성)
ASSESS_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["충족", "부분충족", "미흡", "해당없음"]},
                    "rationale": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
                "required": ["id", "status", "rationale", "recommendation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assessments"],
    "additionalProperties": False,
}


def _call_structured(client, user_msg: str):
    """output_config 구조화 출력으로 호출. SDK가 해당 인자를 모르면 일반 호출로 폴백."""
    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=ASSESS_MAX_TOKENS,
        system=ASSESS_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    try:
        return client.messages.create(
            **kwargs,
            output_config={"format": {"type": "json_schema", "schema": ASSESS_SCHEMA}},
        )
    except TypeError:
        return client.messages.create(**kwargs)  # 구버전 SDK 대비


def assess_category(category: str, situation: str) -> list[dict]:
    """분야 내 각 기준에 대해 충족 여부를 보조 판단. {id,title,status,rationale,recommendation} 리스트 반환.

    API 예외는 호출자에게 전파(friendly_error로 변환 가능). 응답 누락 기준은 '미흡(평가 누락)'으로 채움.
    """
    crit = criteria_in_category(category)
    block = "\n".join(
        f"({c['id']}) {c['title']}: {c['summary']} (점검항목: {c.get('checklist', '')})"
        for c in crit
    )
    user = (
        f"[인증기준 — {category}]\n{block}\n\n"
        f"[회사 현황/증적]\n{situation}\n\n"
        "위 각 기준에 대해 충족 여부를 판단해 JSON으로 출력하세요."
    )

    client = _get_anthropic()
    resp = _call_structured(client, user)
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(_strip_json(text))

    by_id = {c["id"]: c for c in crit}
    parsed = {a.get("id"): a for a in data.get("assessments", []) if isinstance(a, dict)}

    results = []
    for c in crit:  # 분야 기준 순서대로, 누락 방지
        a = parsed.get(c["id"], {})
        status = a.get("status") if a.get("status") in VALID_STATUS else "미흡"
        results.append(
            {
                "id": c["id"],
                "title": c["title"],
                "status": status,
                "rationale": a.get("rationale") or "평가 결과 누락",
                "recommendation": a.get("recommendation") or "",
            }
        )
    return results
