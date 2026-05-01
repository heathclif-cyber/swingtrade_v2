from datetime import datetime, timezone, timedelta
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# WITA (Waktu Indonesia Tengah) = UTC+8
WITA_OFFSET = timedelta(hours=8)
WITA_TZ = timezone(WITA_OFFSET)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def witautcnow() -> datetime:
    """Return current time in WITA (UTC+8) timezone."""
    return datetime.now(WITA_TZ)


def to_wita(dt: datetime | None) -> datetime | None:
    """Convert a UTC datetime to WITA (UTC+8). Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(WITA_TZ)
    return dt.replace(tzinfo=timezone.utc).astimezone(WITA_TZ)


def wita_format(dt: datetime | None, fmt: str = "%m-%d %H:%M") -> str:
    """Convert UTC datetime to WITA and format as string. Returns '—' if None."""
    if dt is None:
        return "—"
    wita_dt = to_wita(dt)
    return wita_dt.strftime(fmt)
