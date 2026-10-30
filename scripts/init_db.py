"""初始化业务库（建表）。

    .venv\\Scripts\\python.exe scripts\\init_db.py

其实服务启动时会自动建表，这个脚本只是让你能单独确认一下数据库是否正常。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.db import engine, init_db  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("init_db")


def main() -> int:
    setup_logging()
    settings.ensure_dirs()
    init_db()

    from sqlalchemy import inspect

    tables = inspect(engine).get_table_names()
    log.info("数据库：%s", settings.database_url)
    log.info("已有表：%s", ", ".join(tables) or "（空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
