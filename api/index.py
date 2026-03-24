"""
Vercel serverless entry point. Vercel's Python runtime detects 'app' as the ASGI entrypoint.

Required env vars in Vercel project settings:
  SUPABASE_URL  — https://scrcfecvmfohrzzwkyaq.supabase.co
  SUPABASE_KEY  — service role key from Supabase dashboard → Settings → API
"""

from pathlib import Path
import os
import sys

# Make the repo root importable (Vercel's working dir is /var/task).
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

os.environ.setdefault("FRONTEND_DIR", str(_repo_root / "frontend"))

from backend.main import app  # noqa: E402 — Vercel detects 'app' as ASGI entrypoint
