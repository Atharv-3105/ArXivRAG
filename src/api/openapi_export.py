"""Write the OpenAPI spec to documentation/api/openapi.json.

    python -m src.api.openapi_export

The spec is generated from the code (FastAPI + Pydantic models), so it cannot drift from the
implementation; committing it lets reviewers see API changes in diffs and clients generate SDKs
without running the server.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "openapi-export-not-a-real-secret-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost:5432/x")

OUT = Path(__file__).resolve().parents[2] / "documentation" / "api" / "openapi.json"


def main() -> None:
    from src.api.app import create_app

    spec = create_app().openapi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(spec['paths'])} paths)")


if __name__ == "__main__":
    main()
