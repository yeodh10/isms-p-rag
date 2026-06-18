"""
config.py — 프로젝트 공통 설정 (경로, 모델명, 상수)

build_index.py / rag.py / app.py 가 이 값을 공유한다.
경로는 이 파일 위치를 기준으로 절대경로화하여, 어디서 실행하든 동일하게 동작한다.
"""

from pathlib import Path

# 프로젝트 루트 (이 파일이 있는 폴더)
BASE_DIR = Path(__file__).parent

# 데이터 / 벡터스토어 경로
DATA_PATH = BASE_DIR / "data" / "isms_criteria.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma"  # .gitignore 로 제외됨 (재생성 가능)

# Chroma 컬렉션 이름
COLLECTION_NAME = "isms_criteria"

# 한국어 임베딩 모델 — 도메인 텍스트가 한국어라 한국어 특화 모델 사용
#   jhgan/ko-sroberta-multitask : KLUE RoBERTa 기반, 한국어 문장 유사도에 강함, 768차원
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 검색 시 가져올 기준 개수 (top-k)
TOP_K = 4

# 답변 생성에 사용할 Claude 모델
CLAUDE_MODEL = "claude-sonnet-4-6"

# 답변 하단 면책 문구 (비공식 참고용 고지)
DISCLAIMER = "본 답변은 비공식 참고용이며, 정확한 내용은 공식 ISMS-P 인증기준(KISA)을 확인하세요."

# 검색 관련성 임계값 — 최상위 결과의 코사인 유사도가 이 값 미만이면
# Claude를 호출하지 않고 "찾지 못했습니다"로 응답한다(오프토픽 질의의 API 비용 차단).
# 보수적으로 낮게 설정: 명백히 무관한 질의만 차단하고 미묘한 경우는 LLM이 판단한다.
SIMILARITY_THRESHOLD = 0.40

# Claude 호출 설정
REQUEST_TIMEOUT = 30   # 초 (응답 지연 시 타임아웃)
MAX_TOKENS = 1500

# 사전 자가점검: 한 분야의 여러 기준을 한 번에 평가하므로 더 큰 출력 한도
ASSESS_MAX_TOKENS = 4000
ASSESS_DISCLAIMER = (
    "이 자가점검 결과는 AI 보조 참고용입니다. 실제 결함 판단과 인증 가부는 KISA 인증심사원/"
    "인증기관이 결정하며, 본 결과는 모의심사·사전 컨설팅을 대체하지 않습니다."
)

# 근거 부족 시 표준 응답
NOT_FOUND_MESSAGE = "해당 내용은 제공된 ISMS-P 기준에서 찾지 못했습니다."

# 세션 단위 레이트리밋 — 공개 앱에서 타인이 API 비용을 남용하지 못하도록 제한
RATE_LIMIT_MAX = 15            # 윈도우당 최대 질문 수
RATE_LIMIT_WINDOW_SEC = 3600   # 윈도우 길이(초)

# 데이터 출처/성격 고지 (README·앱에 표시) — 정직성: 요약문은 공식 원문이 아님
DATA_DISCLAIMER = (
    "각 기준의 점검항목(체크리스트)은 KISA 공식 ISMS-P 2023.11 안내서의 '주요 확인사항'을 "
    "인용했습니다. 다만 요약문(summary)은 비공식 한국어 요약이며 원문이 아니므로, "
    "정확한 문구·해석은 반드시 KISA 공식 문서를 확인하세요."
)
