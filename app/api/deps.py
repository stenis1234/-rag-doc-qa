"""FastAPI 依赖。

统一从 core.db 导出，将来要加「按用户过滤知识库」这类依赖，只需要改这里。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.db import get_db

DbSession = Annotated[Session, Depends(get_db)]
