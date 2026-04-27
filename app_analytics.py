from __future__ import annotations
from database import SessionLocal
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
import json
from pydantic import BaseModel, field_validator
import re
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional


# ─── Request Models ───

class EventPayload(BaseModel):
    """Parsed contents of the nested `payload` JSON string from the client."""

    user_agent: Optional[str] = None
    locale: Optional[str] = None
    location: Optional[str] = None
    referrer: Optional[str] = None
    pathname: Optional[str] = None
    href: Optional[str] = None

    class Config:
        populate_by_name = True


class IncomingEvent(BaseModel):
    """
    The raw analytics event sent by the client via POST /api/analytics.

    The `payload` field arrives as a JSON string containing user-agent, locale,
    location, referrer, pathname, and href. The `parse_payload()` method
    deserializes it into an EventPayload object.
    """

    timestamp: str
    action: str
    version: Optional[str] = None
    session_id: str
    payload: str  # JSON string from the client

    @field_validator("payload", mode="before")
    @classmethod
    def ensure_payload_is_string(cls, v: str | dict) -> str:
        """Accept both a JSON string or a dict; always store as string."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v

    def parse_payload(self) -> EventPayload:
        """Deserialize the payload JSON string into an EventPayload."""
        try:
            raw = json.loads(self.payload)
            return EventPayload(
                user_agent=raw.get("user-agent"),
                locale=raw.get("locale"),
                location=raw.get("location"),
                referrer=raw.get("referrer"),
                pathname=raw.get("pathname"),
                href=raw.get("href"),
            )
        except (json.JSONDecodeError, TypeError):
            return EventPayload()


# ─── Response Models ───


class PageViews(BaseModel):
    pathname: str
    views: int


class CountryVisitors(BaseModel):
    country: str
    visitors: int


class DeviceVisitors(BaseModel):
    device: str
    visitors: int


class DailyVisitors(BaseModel):
    """A single data point for the visitors-over-time line chart."""
    date: str  # ISO date string e.g. "2026-03-27"
    visitors: int


class StatsResponse(BaseModel):
    """Complete analytics stats returned by GET /api/analytics/stats."""

    unique_visitors: int
    total_page_views: int
    views_per_visit: float
    avg_visit_duration_seconds: int
    bounce_rate: float
    visits_per_page: list[PageViews]
    visitors_per_country: list[CountryVisitors]
    visitors_per_device: list[DeviceVisitors]
    visitors_over_time: list[DailyVisitors]


# ─── Device Detection ───
# Regex patterns to classify user-agent strings into device categories.

_TABLET_RE = re.compile(r"tablet|ipad|playbook|silk", re.IGNORECASE)
_MOBILE_RE = re.compile(
    r"mobile|iphone|ipod|android.*mobile|windows phone|blackberry", re.IGNORECASE
)


def detect_device(user_agent: str | None) -> str:
    """Classify a user-agent string as Desktop, Mobile, or Tablet."""
    if not user_agent:
        return "Unknown"
    if _TABLET_RE.search(user_agent):
        return "Tablet"
    if _MOBILE_RE.search(user_agent):
        return "Mobile"
    return "Desktop"


def _parse_iso_timestamp(date_string: str) -> str:
    """
    Normalize an ISO 8601 timestamp string for insertion into a libsql/Turso
    SQLite column. The driver does not accept Python datetime objects as bound
    parameters — bind an ISO 8601 string instead.
    """
    if not date_string or not date_string.strip():
        return datetime.now(timezone.utc).isoformat()

    if 'T' in date_string:
        try:
            # Normalize Z → +00:00, parse for validation, then re-serialize
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            return dt.isoformat()
        except ValueError:
            date_part = date_string.split('T')[0]
            try:
                dt = datetime.strptime(date_part, '%Y-%m-%d')
                return dt.isoformat()
            except ValueError:
                return datetime.now(timezone.utc).isoformat()

    # No T separator — try common date-only formats
    for fmt in ('%Y-%m-%d', '%d-%m-%Y'):
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.isoformat()
        except ValueError:
            continue

    return datetime.now(timezone.utc).isoformat()




# ─── Insert Event ───


async def insert_event(db: AsyncSession, event: IncomingEvent) -> None:
    """
    Parse the incoming event payload and insert a flattened row into the events table.
    The nested JSON payload is exploded into individual columns for efficient querying.
    """
    payload = event.parse_payload()

    # await db.execute(
    db.execute(
        text("""
            INSERT INTO app_user_analytics
                (session_id, action, version, timestamp, user_agent, locale, location, referrer, pathname, href)
            VALUES
                (:session_id, :action, :version, :timestamp, :user_agent, :locale, :location, :referrer, :pathname, :href)
        """),
        {
            "session_id": event.session_id,
            "action": event.action,
            "version": event.version,
            "timestamp": _parse_iso_timestamp(event.timestamp),
            "user_agent": payload.user_agent,
            "locale": payload.locale,
            "location": payload.location,
            "referrer": payload.referrer,
            "pathname": payload.pathname,
            "href": payload.href,
        },
    )
    # await db.commit()
    db.commit()


# Small helper — reuse your existing normalization logic in reverse
def _parse_db_timestamp(value: str) -> datetime:
    """
    Parse a timestamp string returned from libsql/SQLite back into a
    datetime object so arithmetic (subtraction, .total_seconds()) works.
    """
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        # Fallback for date-only values stored without a time component
        return datetime.fromisoformat(value + 'T00:00:00+00:00')


# ─── Get Stats ───


async def get_stats(
    db: AsyncSession,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> StatsResponse:
    """
    Compute all analytics metrics from the events table.
    Supports optional date_from/date_to ISO strings for filtering.

    Returns: unique visitors, page views, views/visit, avg duration,
    bounce rate, visits per page, visitors per country, visitors per device.
    """

    # Build optional WHERE clause for date filtering
    where_clauses: list[str] = []
    params: dict[str, str] = {}

    if date_from:
        where_clauses.append("timestamp >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_clauses.append("timestamp <= :date_to")
        params["date_to"] = date_to

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # ── Unique visitors & total page views ──
    overview = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT
                    COUNT(DISTINCT session_id) AS unique_visitors,
                    COUNT(*)                   AS total_page_views
                FROM app_user_analytics {where}
            """),
            params,
        )
    ).mappings().one()

    unique_visitors: int = overview["unique_visitors"]
    total_page_views: int = overview["total_page_views"]
    views_per_visit = (
        round(total_page_views / unique_visitors, 2) if unique_visitors > 0 else 0.0
    )

    # ── Average visit duration ──
    # Only sessions with 2+ hits have a measurable duration.
    # Duration = time between first and last event in a session.
    duration_rows = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT
                    session_id,
                    MIN(timestamp) AS first_hit,
                    MAX(timestamp) AS last_hit
                FROM app_user_analytics {where}
                GROUP BY session_id
                HAVING COUNT(*) > 1
            """),
            params,
        )
    ).mappings().all()

    avg_visit_duration_seconds = 0
    if duration_rows:
        total_seconds = sum(
            (
                _parse_db_timestamp(row["last_hit"]) - _parse_db_timestamp(row["first_hit"])
            ).total_seconds()
            for row in duration_rows
        )
        avg_visit_duration_seconds = round(total_seconds / len(duration_rows))

    # ── Bounce rate ──
    # A "bounce" is a session with only 1 page view (no navigation).
    bounce_data = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT
                    COUNT(*)                                    AS total_sessions,
                    SUM(CASE WHEN cnt = 1 THEN 1 ELSE 0 END)   AS bounce_sessions
                FROM (
                    SELECT session_id, COUNT(*) AS cnt
                    FROM app_user_analytics {where}
                    GROUP BY session_id
                ) sub
            """),
            params,
        )
    ).mappings().one()

    total_sessions: int = bounce_data["total_sessions"]
    bounce_sessions: int = bounce_data["bounce_sessions"] or 0
    bounce_rate = (
        round((bounce_sessions / total_sessions) * 100, 2) if total_sessions > 0 else 0.0
    )

    # ── Visits per page ──
    page_rows = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT pathname, COUNT(*) AS views
                FROM app_user_analytics {where}
                GROUP BY pathname
                ORDER BY views DESC
            """),
            params,
        )
    ).mappings().all()

    visits_per_page = [
        PageViews(pathname=row["pathname"] or "/", views=row["views"])
        for row in page_rows
    ]

    # ── Visitors per country ──
    country_rows = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT location AS country, COUNT(DISTINCT session_id) AS visitors
                FROM app_user_analytics {where}
                GROUP BY location
                ORDER BY visitors DESC
            """),
            params,
        )
    ).mappings().all()

    visitors_per_country = [
        CountryVisitors(country=row["country"] or "Unknown", visitors=row["visitors"])
        for row in country_rows
    ]

    # ── Visitors per device ──
    # Group user-agents by session, classify each into Desktop/Mobile/Tablet
    ua_rows = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT session_id, user_agent
                FROM app_user_analytics {where}
                GROUP BY session_id, user_agent
            """),
            params,
        )
    ).mappings().all()

    device_map: dict[str, set[str]] = {}
    for row in ua_rows:
        device = detect_device(row["user_agent"])
        device_map.setdefault(device, set()).add(row["session_id"])

    visitors_per_device = sorted(
        [
            DeviceVisitors(device=device, visitors=len(sessions))
            for device, sessions in device_map.items()
        ],
        key=lambda d: d.visitors,
        reverse=True,
    )

    # ── Visitors over time (daily) ──
    # Groups unique sessions by calendar date for the line chart.
    daily_rows = (
        # await db.execute(
        db.execute(
            text(f"""
                SELECT
                    DATE(timestamp) AS visit_date,
                    COUNT(DISTINCT session_id) AS visitors
                FROM app_user_analytics {where}
                GROUP BY DATE(timestamp)
                ORDER BY visit_date ASC
            """),
            params,
        )
    ).mappings().all()

    visitors_over_time = [
        DailyVisitors(date=str(row["visit_date"]), visitors=row["visitors"])
        for row in daily_rows
    ]

    return StatsResponse(
        unique_visitors=unique_visitors,
        total_page_views=total_page_views,
        views_per_visit=views_per_visit,
        avg_visit_duration_seconds=avg_visit_duration_seconds,
        bounce_rate=bounce_rate,
        visits_per_page=visits_per_page,
        visitors_per_country=visitors_per_country,
        visitors_per_device=visitors_per_device,
        visitors_over_time=visitors_over_time,
    )

# ─── Router ───

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Routes ───


@router.post("/api/analytics", status_code=200)
async def post_event(
    event: IncomingEvent,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Ingest a single analytics event. Parses payload and inserts into PostgreSQL."""
    await insert_event(db, event)
    return {"success": True}


@router.get("/api/analytics/stats", response_model=StatsResponse)
async def get_analytics_stats(
    db: AsyncSession = Depends(get_db),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> StatsResponse:
    """Return computed analytics stats. Supports optional date_from/date_to filters."""
    return await get_stats(db, date_from=date_from, date_to=date_to)


@router.get("/health")
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     migration_path = Path(__file__).parent / "migrations" / "001_create_events.sql"
#     migration_sql = migration_path.read_text(encoding="utf-8")

#     async with engine.begin() as conn:
#         for statement in migration_sql.split(";"):
#             statement = statement.strip()
#             if statement:
#                 await conn.execute(text(statement))

#     yield

#     await engine.dispose()


# # ─── App Factory ───
# def create_app() -> FastAPI:
#     app = FastAPI(
#         title="Analytics API",
#         description="Simple analytics backend for your webapp",
#         version="1.0.0",
#         # lifespan=lifespan,
#     )

#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     app.include_router(router)
#     return app

# # ─── Main Entry ───
# if __name__ == "__main__":
#     app = create_app()
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


