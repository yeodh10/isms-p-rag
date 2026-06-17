"""pytest 설정: 저장소 루트를 import 경로에 추가해 tests/에서 rag·config를 import 가능하게 함."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
