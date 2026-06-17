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
