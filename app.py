"""
app.py — ISMS-P 인증기준 RAG 챗봇 Streamlit UI

기능:
  - 대화형 챗봇 UI (멀티턴 히스토리) + 스트리밍 답변
  - 비밀번호 게이트 + 세션 레이트리밋 (공개 앱의 API 비용 남용 방지)
  - 출처 인용 + 참고한 기준 표시 + 비공식/데이터 출처 고지

실행: streamlit run app.py

설계 메모:
  각 질문은 독립적으로 검색·근거화한다(이전 대화를 LLM에 넘기지 않음). 컴플라이언스
  맥락에서 답변이 항상 '검색된 기준'에만 근거하도록 보장하기 위함이며, 화면에는 대화
  히스토리로 표시한다.
"""

import hmac
import os
import time

import streamlit as st

from build_index import ensure_index
from config import (
    DATA_DISCLAIMER,
    DISCLAIMER,
    EMBED_MODEL_NAME,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW_SEC,
    TOP_K,
)
import rag

# 페이지 설정 (가장 먼저)
st.set_page_config(page_title="ISMS-P 인증기준 도우미", page_icon="🔐", layout="centered")


def _inject_secret(name: str) -> None:
    """배포(Streamlit Cloud) 시 secrets를 환경변수로 주입. 로컬은 .env(dotenv) 사용."""
    try:
        if name in st.secrets and not os.environ.get(name):
            os.environ[name] = str(st.secrets[name])
    except Exception:
        pass  # secrets.toml 없는 로컬 실행


_inject_secret("ANTHROPIC_API_KEY")


# ── 비밀번호 게이트 (APP_PASSWORD 가 설정된 경우에만 활성) ──────────
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
        return True  # 게이트 미설정 → 공개 접근 허용
    if st.session_state.get("authed"):
        return True
    st.title("🔐 ISMS-P 인증기준 RAG 챗봇")
    st.info("이 앱은 비밀번호로 보호됩니다. 접속 비밀번호를 입력하세요.")
    entered = st.text_input("접속 비밀번호", type="password")
    if entered:
        if hmac.compare_digest(entered.encode("utf-8"), pw.encode("utf-8")):
            st.session_state.authed = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")
    return False


# ── 세션 레이트리밋 ────────────────────────────────────────────
def _recent_q_times() -> list:
    now = time.time()
    return [t for t in st.session_state.get("q_times", []) if now - t < RATE_LIMIT_WINDOW_SEC]


def rate_limited() -> bool:
    return len(_recent_q_times()) >= RATE_LIMIT_MAX


def record_question() -> None:
    st.session_state.q_times = _recent_q_times() + [time.time()]


# ── 인덱스 준비 (배포 첫 실행 시 자동 빌드) ──────────────────────
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


# ── 메인 ───────────────────────────────────────────────────────
if not require_auth():
    st.stop()

_prepare_index()

st.info(
    "ℹ️ 본 도구는 **비공식 참고용**입니다. 공식 인증기준은 "
    "[KISA ISMS-P 자료실](https://isms.kisa.or.kr)에서 확인하세요."
)
st.title("🔐 ISMS-P 인증기준 RAG 챗봇")
st.caption(
    "ISMS-P 인증기준(101개)에 대해 질문하면, 관련 기준을 검색해 쉬운 말로 답하고 "
    "근거 기준 번호를 출처로 인용합니다."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리 렌더
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
            st.caption(f"⚠️ {DISCLAIMER}")

# 예시 질문 (대화 시작 전에만 노출)
clicked = None
if not st.session_state.messages:
    st.write("**예시 질문**")
    cols = st.columns(3)
    examples = ["접근통제 관련 기준을 알려주세요", "개인정보 파기에 대한 요구사항은?", "위험평가는 어느 기준?"]
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            clicked = ex

prompt = st.chat_input("질문을 입력하세요") or clicked

if prompt:
    if rate_limited():
        st.warning(f"질문이 너무 많습니다(시간당 {RATE_LIMIT_MAX}개 제한). 잠시 후 다시 시도하세요.")
        st.stop()
    record_question()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 ISMS-P 기준을 검색하는 중..."):
                sources = rag.retrieve(prompt, k=TOP_K)
        except Exception as e:  # noqa: BLE001 - 인덱스 없음 등
            st.error("검색 중 오류가 발생했습니다. 인덱스가 생성되었는지 확인하세요 (`python build_index.py`).")
            with st.expander("오류 상세"):
                st.code(str(e))
            st.stop()

        try:
            answer_text = st.write_stream(rag.stream_answer(prompt, hits=sources))
        except Exception as e:  # noqa: BLE001 - API 오류, 검색 결과는 보존
            answer_text = f"⚠️ {rag.friendly_error(e)}"
            st.error(answer_text)

        render_sources(sources)
        st.caption(f"⚠️ {DISCLAIMER}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )

# 사이드바
with st.sidebar:
    st.header("ℹ️ 정보")
    st.markdown(
        "**ISMS-P 인증기준 RAG 챗봇**\n\n"
        "- 기준: ISMS-P 인증기준 **101개** (2023.11)\n"
        "  - 관리체계 수립 및 운영 16\n"
        "  - 보호대책 요구사항 64\n"
        "  - 개인정보 처리단계별 요구사항 21\n"
        f"- 검색: 한국어 임베딩(`{EMBED_MODEL_NAME}`) + Chroma\n"
        "- 답변: Anthropic Claude (스트리밍)\n\n"
        "답변은 검색된 기준에만 근거하며, 근거가 없으면 "
        '"찾지 못했습니다"라고 답합니다(환각 억제).'
    )
    st.divider()
    st.caption(f"📌 {DATA_DISCLAIMER}")
    if st.session_state.get("messages") and st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
