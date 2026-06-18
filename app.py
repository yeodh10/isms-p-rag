"""
app.py — ISMS-P 인증기준 RAG 챗봇 + 사전 자가점검 Streamlit UI

두 가지 모드 (사이드바에서 전환):
  💬 질문하기      — 기준 검색 + 출처 인용 답변(스트리밍) 챗봇
  ✅ 사전 자가점검 — 분야별 현황을 입력하면 기준별 충족 여부를 AI가 보조 판단 + 갭 리포트

공통: 비밀번호 게이트 + 세션 레이트리밋(공개 앱 API 비용 보호), 비공식/데이터 출처 고지.

실행: streamlit run app.py
"""

import hmac
import os
import time
from collections import Counter

import streamlit as st

import assess
import docparse
import rag
from build_index import ensure_index
from config import (
    ASSESS_DISCLAIMER,
    DATA_DISCLAIMER,
    DISCLAIMER,
    EMBED_MODEL_NAME,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW_SEC,
    TOP_K,
)

_CSS = """<style>
[data-testid="stToolbar"], #MainMenu, footer {visibility: hidden; height: 0;}
.block-container {padding-top: 2.2rem;}
.isms-hero {background: linear-gradient(120deg, #4338ca 0%, #312e81 45%, #1e293b 100%);
  border-radius: 14px; padding: 20px 26px; margin-bottom: 10px;
  box-shadow: 0 8px 24px rgba(49,46,129,0.35);}
.isms-hero .t {color: #ffffff; font-size: 1.6rem; font-weight: 700;}
.isms-hero .s {color: #c7d2fe; font-size: 0.93rem; margin-top: 6px;}
div.stButton > button {border-radius: 10px;}
[data-testid="stMetric"] {background: #f8fafc;
  border: 1px solid #e2e8f0; border-radius: 12px; padding: 8px 12px;}
</style>"""

_HERO = (
    '<div class="isms-hero"><div class="t">🔐 ISMS-P 인증기준 도우미</div>'
    '<div class="s">인증기준 질의응답 + 사전 자가점검 · AI 보조 · 비공식 참고용</div></div>'
)

st.set_page_config(page_title="ISMS-P 인증기준 도우미", page_icon="🔐", layout="centered")
st.markdown(_CSS, unsafe_allow_html=True)


def _inject_secret(name: str) -> None:
    try:
        if name in st.secrets and not os.environ.get(name):
            os.environ[name] = str(st.secrets[name])
    except Exception:
        pass


_inject_secret("ANTHROPIC_API_KEY")


# ── 비밀번호 게이트 ─────────────────────────────────────────────
def _app_password():
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD")


def require_auth() -> bool:
    pw = _app_password()
    if not pw:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("🔐 ISMS-P 인증기준 도우미")
    st.info("이 앱은 비밀번호로 보호됩니다. 접속 비밀번호를 입력하세요.")
    entered = st.text_input("접속 비밀번호", type="password")
    if entered:
        if hmac.compare_digest(entered.encode("utf-8"), pw.encode("utf-8")):
            st.session_state.authed = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")
    return False


# ── 세션 레이트리밋 (모드 공유) ─────────────────────────────────
def _recent_q_times() -> list:
    now = time.time()
    return [t for t in st.session_state.get("q_times", []) if now - t < RATE_LIMIT_WINDOW_SEC]


def rate_limited() -> bool:
    return len(_recent_q_times()) >= RATE_LIMIT_MAX


def record_question() -> None:
    st.session_state.q_times = _recent_q_times() + [time.time()]


@st.cache_resource(show_spinner="최초 실행: 인증기준 색인을 준비하는 중입니다... (최대 1~2분)")
def _prepare_index():
    ensure_index()
    return True


def render_sources(sources: list) -> None:
    with st.expander(f"📚 참고한 기준 {len(sources)}개 보기"):
        for h in sources:
            st.markdown(
                f"**({h['id']}) {h.get('title','')}**  ·  {h.get('category','')}"
                f"  ·  유사도 {h.get('similarity', 0):.3f}"
            )
            st.write(h.get("summary", ""))
            if h.get("checklist"):
                st.caption(f"점검항목 예시: {h['checklist']}")


# ── 모드 1: 질문하기 (챗봇) ─────────────────────────────────────
def render_chat() -> None:
    st.caption(
        "ISMS-P 인증기준(101개)에 대해 질문하면, 관련 기준을 검색해 쉬운 말로 답하고 "
        "근거 기준 번호를 출처로 인용합니다."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
                st.caption(f"⚠️ {DISCLAIMER}")

    clicked = None
    if not st.session_state.messages:
        st.write("**예시 질문**")
        examples = [
            "접근통제 관련 기준을 알려주세요",
            "개인정보 파기에 대한 요구사항은?",
            "위험평가는 어느 기준?",
            "비밀번호는 어떻게 관리하나요?",
        ]
        for i in range(0, len(examples), 2):
            cols = st.columns(2)
            for col, ex in zip(cols, examples[i : i + 2]):
                if col.button(ex, use_container_width=True):
                    clicked = ex

    prompt = st.chat_input("질문을 입력하세요") or clicked
    if not prompt:
        return

    if rate_limited():
        st.warning(f"질문이 너무 많습니다(시간당 {RATE_LIMIT_MAX}개 제한). 잠시 후 다시 시도하세요.")
        return
    record_question()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 ISMS-P 기준을 검색하는 중..."):
                sources = rag.retrieve(prompt, k=TOP_K)
        except Exception as e:  # noqa: BLE001
            st.error("검색 중 오류가 발생했습니다. 인덱스가 생성되었는지 확인하세요 (`python build_index.py`).")
            with st.expander("오류 상세"):
                st.code(str(e))
            return

        try:
            answer_text = st.write_stream(rag.stream_answer(prompt, hits=sources))
        except Exception as e:  # noqa: BLE001
            answer_text = f"⚠️ {rag.friendly_error(e)}"
            st.error(answer_text)

        render_sources(sources)
        st.caption(f"⚠️ {DISCLAIMER}")

    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": sources})


# ── 모드 2: 사전 자가점검 (AI 보조 갭분석) ──────────────────────
_BADGE = {"충족": "✅ 충족", "부분충족": "🟡 부분충족", "미흡": "❌ 미흡", "해당없음": "⚪ 해당없음"}
_PILL = {"충족": "#16a34a", "부분충족": "#d97706", "미흡": "#dc2626", "해당없음": "#6b7280"}


def _pill(status: str) -> str:
    c = _PILL.get(status, "#6b7280")
    return (
        f'<span style="background:{c}22;color:{c};border:1px solid {c}66;'
        f'padding:2px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">{status}</span>'
    )


def _build_report(category: str, results: list, counts: Counter) -> str:
    lines = [
        f"# ISMS-P 사전 자가점검 결과 — {category}",
        "",
        "> AI 보조 참고용 · 비공식. 결함 판단·인증 가부는 KISA 심사원/인증기관이 결정합니다.",
        "",
        "## 요약",
        f"- 충족 {counts.get('충족', 0)} · 부분충족 {counts.get('부분충족', 0)} · "
        f"미흡 {counts.get('미흡', 0)} · 해당없음 {counts.get('해당없음', 0)}",
        "",
        "## 기준별 결과",
        "",
    ]
    for r in results:
        lines.append(f"### ({r['id']}) {r['title']} — {r['status']}")
        lines.append(f"- 근거: {r['rationale']}")
        if r["recommendation"]:
            lines.append(f"- 보완 권고: {r['recommendation']}")
        lines.append("")
    return "\n".join(lines)


def _render_results(category: str, results: list) -> None:
    counts = Counter(r["status"] for r in results)
    st.subheader(f"📋 자가점검 결과 — {category}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ 충족", counts.get("충족", 0))
    c2.metric("🟡 부분충족", counts.get("부분충족", 0))
    c3.metric("❌ 미흡", counts.get("미흡", 0))
    c4.metric("⚪ 해당없음", counts.get("해당없음", 0))

    for r in results:
        st.markdown(
            f"{_pill(r['status'])}&nbsp; <b>({r['id']}) {r['title']}</b>",
            unsafe_allow_html=True,
        )
        st.caption(f"근거: {r['rationale']}")
        if r["recommendation"]:
            st.caption(f"보완 권고: {r['recommendation']}")
    st.caption(f"⚠️ {ASSESS_DISCLAIMER}")
    st.download_button(
        "📥 이 분야 갭 리포트 (.md)",
        _build_report(category, results, counts),
        file_name=f"자가점검_{category}.md",
        mime="text/markdown",
        key=f"dl_{category}",
    )


def _build_combined_report(assessments: dict, total: Counter) -> str:
    lines = [
        "# ISMS-P 사전 자가점검 — 통합 갭 리포트",
        "",
        "> AI 보조 참고용 · 비공식. 결함 판단·인증 가부는 KISA 심사원/인증기관이 결정합니다.",
        "",
        "## 전체 요약",
        f"- 점검한 분야: {len(assessments)}개",
        f"- 충족 {total.get('충족', 0)} · 부분충족 {total.get('부분충족', 0)} · "
        f"미흡 {total.get('미흡', 0)} · 해당없음 {total.get('해당없음', 0)}",
        "",
    ]
    for cat, results in assessments.items():
        counts = Counter(r["status"] for r in results)
        lines.append(f"## {cat}")
        lines.append(
            f"- 충족 {counts.get('충족', 0)} · 부분충족 {counts.get('부분충족', 0)} · "
            f"미흡 {counts.get('미흡', 0)} · 해당없음 {counts.get('해당없음', 0)}"
        )
        lines.append("")
        for r in results:
            lines.append(f"### ({r['id']}) {r['title']} — {r['status']}")
            lines.append(f"- 근거: {r['rationale']}")
            if r["recommendation"]:
                lines.append(f"- 보완 권고: {r['recommendation']}")
            lines.append("")
    return "\n".join(lines)


def _render_cumulative(scope: str, assessments: dict) -> None:
    st.divider()
    st.subheader("📊 누적 갭 분석 (점검한 모든 분야 합산)")
    total = Counter()
    for results in assessments.values():
        total.update(r["status"] for r in results)

    all_cats = [c for _dom, c in assess.categories(scope)]
    done = [c for c in all_cats if c in assessments]
    st.progress(
        min(len(done) / len(all_cats), 1.0) if all_cats else 0.0,
        text=f"점검 완료: {len(done)} / {len(all_cats)} 분야",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ 충족", total.get("충족", 0))
    c2.metric("🟡 부분충족", total.get("부분충족", 0))
    c3.metric("❌ 미흡", total.get("미흡", 0))
    c4.metric("⚪ 해당없음", total.get("해당없음", 0))

    remaining = [c for c in all_cats if c not in assessments]
    if remaining:
        with st.expander(f"아직 점검하지 않은 분야 {len(remaining)}개"):
            st.write(" · ".join(remaining))

    st.download_button(
        "📥 통합 갭 리포트 내려받기 (.md)",
        _build_combined_report(assessments, total),
        file_name="자가점검_통합리포트.md",
        mime="text/markdown",
        key="dl_combined",
    )
    if st.button("누적 결과 초기화", key="reset_assessments"):
        st.session_state.assessments = {}
        st.rerun()


def render_assessment() -> None:
    st.caption(
        "분야를 고르고 회사의 현황·증적을 입력하면, 해당 분야 기준별로 충족 여부를 AI가 "
        "보조 판단하고 보완점을 제시합니다."
    )
    st.warning(f"⚠️ {ASSESS_DISCLAIMER}")

    scope_label = st.radio("인증 범위", ["ISMS-P (101개)", "ISMS (80개)"], horizontal=True)
    scope = "ISMS-P" if scope_label.startswith("ISMS-P") else "ISMS"

    cats = assess.categories(scope)
    labels = [f"{dom}  ›  {cat}" for dom, cat in cats]
    sel = st.selectbox("점검할 분야", labels)
    chosen_cat = cats[labels.index(sel)][1]

    crit = assess.criteria_in_category(chosen_cat)
    with st.expander(f"이 분야 기준 {len(crit)}개 — 무엇을 점검하나"):
        for c in crit:
            st.markdown(f"**({c['id']}) {c['title']}**")
            st.caption(c.get("checklist") or c.get("summary", ""))

    uploaded = st.file_uploader(
        "문서 업로드 (PDF·DOCX·TXT) — 선택. 업로드하면 내용을 추출해 아래 입력창에 채웁니다.",
        type=["pdf", "docx", "txt", "md"],
    )
    if uploaded is not None and st.session_state.get("_last_file") != uploaded.name:
        try:
            st.session_state.situation_input = docparse.extract_text(uploaded.name, uploaded.getvalue())
            st.session_state._last_file = uploaded.name
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"문서 텍스트 추출 실패: {e}")

    situation = st.text_area(
        "우리 회사 현황·증적 (직접 붙여넣거나, 위에서 문서 업로드)",
        height=200,
        placeholder=(
            "예) 비밀번호는 9자 이상·분기 1회 변경 강제, 5회 실패 시 계정 잠금. "
            "관리자 계정은 별도 식별 후 MFA 적용. 접근권한은 팀장 승인 후 부여하고 반기 1회 검토함..."
        ),
        key="situation_input",
    )

    if st.button("자가점검 실행", type="primary"):
        if rate_limited():
            st.warning(f"요청이 너무 많습니다(시간당 {RATE_LIMIT_MAX}회 제한). 잠시 후 다시 시도하세요.")
        elif not situation.strip():
            st.warning("회사 현황·증적을 입력하세요.")
        else:
            record_question()
            try:
                with st.spinner("기준별 충족 여부를 분석하는 중..."):
                    results = assess.assess_category(chosen_cat, situation)
                st.session_state.setdefault("assessments", {})[chosen_cat] = results
            except Exception as e:  # noqa: BLE001
                st.error(f"⚠️ {rag.friendly_error(e)}")

    assessments = st.session_state.get("assessments", {})
    if chosen_cat in assessments:
        _render_results(chosen_cat, assessments[chosen_cat])
    if assessments:
        _render_cumulative(scope, assessments)


# ── 메인 ───────────────────────────────────────────────────────
if not require_auth():
    st.stop()

_prepare_index()

st.markdown(_HERO, unsafe_allow_html=True)
st.caption("ⓘ 공식 인증기준은 [KISA ISMS-P 자료실](https://isms.kisa.or.kr)에서 확인하세요.")

with st.sidebar:
    mode = st.radio("모드", ["💬 질문하기", "✅ 사전 자가점검"])
    st.divider()
    st.header("ℹ️ 정보")
    st.markdown(
        "- 기준: ISMS-P **101개** (2023.11) — 관리체계 16 / 보호대책 64 / 개인정보 21\n"
        f"- 검색: 한국어 임베딩(`{EMBED_MODEL_NAME}`) + Chroma\n"
        "- 답변: Anthropic Claude (스트리밍)\n\n"
        "답변은 검색된 기준에만 근거하며, 근거가 없으면 \"찾지 못했습니다\"라고 답합니다(환각 억제)."
    )
    st.divider()
    st.caption(f"📌 {DATA_DISCLAIMER}")
    if st.session_state.get("messages") and st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if mode.startswith("💬"):
    render_chat()
else:
    render_assessment()
