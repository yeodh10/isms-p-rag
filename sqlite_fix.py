"""
sqlite_fix.py — chromadb는 sqlite3 >= 3.35 가 필요하다.

일부 배포 환경(예: Streamlit Community Cloud)은 구버전 sqlite3를 제공하므로,
pysqlite3-binary 가 설치되어 있으면 표준 sqlite3 대신 사용하도록 교체한다.
※ chromadb 를 import 하기 전에 가장 먼저 import 되어야 한다.
"""

try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # 로컬(Windows 등) pysqlite3 미설치 환경은 시스템 sqlite3 를 그대로 사용
