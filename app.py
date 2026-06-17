"""
app.py — ISMS-P 인증기준 RAG 챗봇 Streamlit UI (Phase 4)

실행:
  streamlit run app.py

구성:
  - 상단 고지문 (비공식 참고용)
  - 질문 입력 + 예시 질문 버튼
  - 답변 영역 (출처 기준 번호 인용)
  - "참고한 기준" 펼치기 (검색된 기준 원문 표시)

설계 메모:
  검색(retrieve)과 답변(answer)을 분리 처리한다. Claude 호출이 실패해도
  (예: API 크레딧 부족) 검색 결과는 항상 표시되어 앱이 죽지 않는다.
"""

import os

import streamlit as st

from config import DISCLAIMER, EMBED_MODEL_NAME, TOP_K
from rag import answer, retrieve

# 배포(Streamlit Cloud) 시 secrets를 환경변수로 주입 — 로컬은 .env(dotenv)를 사용
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", str(st.secrets["ANTHROPIC_API_KEY"]))
except Exception:
    pass  # secrets.toml 이 없으면(로컬 실행) 무시하고 .env 를 사용

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISMS-P 인증기준 도우미",
    page_icon="🔐",
    layout="centered",
)

# 상단 고지문
st.info(
    "ℹ️ 본 도구는 **비공식 참고용**입니다. 공식 인증기준은 "
    "[KISA ISMS-P 자료실](https://isms.kisa.or.kr)에서 확인하세요."
)

st.title("🔐 ISMS-P 인증기준 RAG 챗봇")
st.caption(
    "ISMS-P 인증기준(101개)에 대해 질문하면, 관련 기준을 검색해 쉬운 말로 답하고 "
    "근거 기준 번호를 출처로 인용합니다."
)

# ─────────────────────────────────────────────────────────────
# 예시 질문 버튼
# ─────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "접근통제 관련 기준을 알려주세요",
    "개인정보 파기에 대한 요구사항은?",
    "위험평가는 어느 기준에 해당하나요?",
]

st.write("**예시 질문**")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
clicked_example = None
for col, ex in zip(example_cols, EXAMPLE_QUESTIONS):
    if col.button(ex, use_container_width=True):
        clicked_example = ex

# ─────────────────────────────────────────────────────────────
# 질문 입력 폼
# ─────────────────────────────────────────────────────────────
with st.form("ask_form"):
    typed_question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 비밀번호는 어떻게 관리해야 하나요?",
    )
    submitted = st.form_submit_button("질문하기", type="primary")

# 처리할 질문 결정: 예시 버튼 클릭이 우선, 없으면 폼 제출
question = clicked_example or (typed_question if submitted else None)


# ─────────────────────────────────────────────────────────────
# 질문 처리 및 결과 표시
# ─────────────────────────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    st.subheader("📚 참고한 기준")
    for h in sources:
        header = (
            f"({h['id']}) {h.get('title', '')}  ·  {h.get('category', '')}  "
            f"·  유사도 {h.get('similarity', 0):.3f}"
        )
        with st.expander(header):
            st.write(h.get("summary", ""))
            if h.get("checklist"):
                st.caption(f"점검항목 예시: {h['checklist']}")


if question is not None and question.strip():
    # 1) 검색 — 항상 동작 (API 불필요)
    try:
        with st.spinner("관련 ISMS-P 기준을 검색하는 중..."):
            sources = retrieve(question, k=TOP_K)
    except Exception as e:  # noqa: BLE001
        st.error(
            "⚠️ 검색 중 오류가 발생했습니다. 벡터 인덱스가 생성되었는지 확인하세요 "
            "(`python build_index.py` 실행)."
        )
        with st.expander("오류 상세"):
            st.code(str(e))
        st.stop()

    # 2) 답변 생성 — Claude 호출 (크레딧 필요), 실패해도 검색 결과는 보존
    answer_text, answer_error = None, None
    with st.spinner("Claude로 답변을 생성하는 중..."):
        try:
            answer_text = answer(question, hits=sources)["answer"]
        except Exception as e:  # noqa: BLE001 - UI에서는 모든 예외를 친절히 처리
            answer_error = e

    # 3) 답변 표시
    st.subheader("💬 답변")
    if answer_text:
        st.markdown(answer_text)
        st.caption(f"⚠️ {DISCLAIMER}")
    else:
        msg = str(answer_error)
        if "credit balance" in msg.lower():
            st.warning(
                "⚠️ Anthropic API 크레딧 잔액이 부족하여 답변을 생성하지 못했습니다.\n\n"
                "https://console.anthropic.com → **Plans & Billing** 에서 크레딧을 충전하세요. "
                "아래 **참고한 기준**은 정상적으로 검색되었습니다."
            )
        elif "api_key" in msg.lower() or "authentication" in msg.lower():
            st.error("⚠️ ANTHROPIC_API_KEY 설정에 문제가 있습니다. .env 파일을 확인하세요.")
        else:
            st.error("⚠️ 답변 생성 중 오류가 발생했습니다. 아래 참고한 기준은 정상 검색되었습니다.")
            with st.expander("오류 상세"):
                st.code(msg)

    # 4) 참고한 기준 — 항상 표시
    render_sources(sources)

elif question is not None:
    # 빈 질문 제출
    st.warning("질문을 입력해주세요.")

# ─────────────────────────────────────────────────────────────
# 사이드바 — 프로젝트 정보
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ 정보")
    st.markdown(
        "**ISMS-P 인증기준 RAG 챗봇**\n\n"
        "- 기준: ISMS-P 인증기준 **101개** (2023.11)\n"
        "  - 관리체계 수립 및 운영 16\n"
        "  - 보호대책 요구사항 64\n"
        "  - 개인정보 처리단계별 요구사항 21\n"
        f"- 검색: 한국어 임베딩(`{EMBED_MODEL_NAME}`) + Chroma\n"
        "- 답변: Anthropic Claude\n\n"
        "답변은 검색된 기준에만 근거하며, 근거가 없으면 "
        "\"찾지 못했습니다\"라고 답합니다(환각 억제)."
    )
    st.divider()
    st.caption("⚠️ 비공식 참고용 · 공식 기준은 KISA 확인")
