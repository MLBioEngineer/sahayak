"""
Supabase client helper for Sahayak.
Handles conversation logging and persistence if SUPABASE_URL and SUPABASE_KEY are provided.
Fails gracefully if credentials are not configured or temporarily unreachable.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("sahayak.db_service")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

_supabase_client = None


def get_supabase():
    """Initializes and caches the Supabase client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("Supabase credentials not configured in environment; database persistence is disabled.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None


def log_message(session_id: str, role: str, content: str):
    """Logs a message to the 'chat_history' table in Supabase if configured."""
    client = get_supabase()
    if not client:
        return

    try:
        client.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        logger.warning(f"Could not persist message to Supabase: {e}")
