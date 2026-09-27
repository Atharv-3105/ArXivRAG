"""Prompt A/B report from production data: the messages table (ADR 0010, 0012).

    python -m src.eval.ab [--days 14]

Per prompt_version: answers, check pass rates, guardrail approval, and - for tutor grading -
how often learners got the next question right. Compare variants only with enough answers each;
the report prints n so a 3-answer "win" is not mistaken for a result.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

_AB = text(
    """
    SELECT prompt_version,
           count(*) AS answers,
           avg(CASE WHEN (checks->>'guardrail_approved')::boolean THEN 1.0 ELSE 0.0 END)
               AS guardrail_approved,
           avg(CASE WHEN (checks->>'one_question')::boolean THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE checks ? 'one_question') AS one_question,
           avg(CASE WHEN (checks->>'trailer_parsed')::boolean THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE checks ? 'trailer_parsed') AS trailer_parsed,
           avg(CASE WHEN (checks->>'citations_grounded')::boolean THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE checks ? 'citations_grounded') AS citations_grounded
    FROM messages
    WHERE role = 'assistant' AND prompt_version IS NOT NULL AND created_at >= :since
    GROUP BY prompt_version
    ORDER BY prompt_version
    """
)


async def report(days: int) -> list[dict]:
    from src.storage.database import dispose_engine, get_async_session

    since = datetime.now(UTC) - timedelta(days=days)
    try:
        async with get_async_session() as session:
            rows = (await session.execute(_AB, {"since": since})).mappings().all()
        # avg() returns Decimal; counts and labels stay as they are
        return [
            {
                k: v if v is None or isinstance(v, str | int) else round(float(v), 4)
                for k, v in row.items()
            }
            for row in rows
        ]
    finally:
        await dispose_engine()


def main() -> None:
    cli = argparse.ArgumentParser(description="Prompt variant comparison from stored answers")
    cli.add_argument("--days", type=int, default=14)
    args = cli.parse_args()
    print(json.dumps(asyncio.run(report(args.days)), indent=2, default=str))


if __name__ == "__main__":
    main()
