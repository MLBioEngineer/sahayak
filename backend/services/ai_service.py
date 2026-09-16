"""
Sahayak AI Inference Service.
Connects to a self-hosted Ollama instance (e.g. on Hugging Face Spaces, Cloudflare Tunnel, or VPS).
Implements resilient timeout handling, Bengali medical system prompt, and graceful fallbacks.
"""
import os
import logging
import httpx

logger = logging.getLogger("sahayak.ai_service")

# Configuration from environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("MODEL_ENDPOINT", "")).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0"))

# Bengali Medical System Prompt
SYSTEM_PROMPT = (
    "আপনি 'সহায়ক' (Sahayak) — একটি সহানুভূতিশীল, পেশাদার এবং নির্ভরযোগ্য বাংলা মেডিকেল এআই সহকারী। "
    "ব্যবহারকারীর স্বাস্থ্য সমস্যা ও লক্ষণগুলো মনোযোগ দিয়ে শুনুন এবং বাংলায় স্পষ্ট, সহজ ও ব্যবহারোপযোগী প্রাথমিক পরামর্শ দিন। "
    "গুরুত্বপূর্ণ নির্দেশনা:\n"
    "১. সর্বদা খাঁটি ও প্রাঞ্জল বাংলায় উত্তর দিন।\n"
    "২. কখনোই ক্ষতিকর বা অপ্রমাণিত প্রেসক্রিপশন দেবেন না।\n"
    "৩. গুরুতর, দীর্ঘস্থায়ী বা তীব্র লক্ষণের ক্ষেত্রে অবিলম্বে নিকটস্থ হাসপাতাল অথবা রেজিস্টার্ড চিকিৎসকের কাছে যাওয়ার স্পষ্ট পরামর্শ দিন।\n"
    "৪. রোগীর মানসিক স্বস্তির জন্য সহানুভূতিশীল ভাষা ব্যবহার করুন।"
)

# Resilient Bengali fallback messages when LLM is cold-starting or temporarily unreachable
FALLBACK_REPLY = (
    "দুঃখিত, সহায়ক এআই ইঞ্জিন বর্তমানে কিছুটা ব্যস্ত রয়েছে অথবা মডেলটি চালু হচ্ছে। "
    "অনুগ্রহ করে এক মিনিট পর আবার বার্তা পাঠান।\n\n"
    "জরুরি শারীরিক সমস্যা বা মারাত্মক উপসর্গের ক্ষেত্রে অনুগ্রহ করে বিলম্ব না করে নিকটস্থ হাসপাতালে যান "
    "অথবা জাতীয় জরুরি সেবা ৯৯৯ এ যোগাযোগ করুন।"
)

DEMO_RESPONSES = [
    "এটি সাধারণ স্বাস্থ্য সুরক্ষার প্রাথমিক পরামর্শ: পর্যাপ্ত পানি পান করুন, পুষ্টিকর খাবার গ্রহণ করুন এবং বিশ্রাম নিন। লক্ষণ ৩ দিনের বেশি স্থায়ী হলে বা তীব্র আকার ধারণ করলে নিকটস্থ চিকিৎসকের পরামর্শ নিন।",
    "আপনার শারীরিক অবস্থা পর্যবেক্ষণে রাখা জরুরি। যদি জ্বর, শ্বাসকষ্ট বা অস্বাভাবিক ব্যথা থাকে তবে অবহেলা না করে একজন রেজিস্টার্ড ডাক্তারের কাছে স্বাস্থ্য পরীক্ষা করান।",
    "সহায়ক এআই ইঞ্জিনের প্রাথমিক সংস্করণ সক্রিয় রয়েছে। আপনি কেমন বোধ করছেন তা বিস্তারিত জানালে প্রাথমিক স্বাস্থ্য সচেতনতামূলক তথ্য দেওয়া সম্ভব।"
]


def get_ai_response(user_message: str, chat_history: list[dict] | None = None) -> str:
    """
    Generate an AI response using the self-hosted Ollama endpoint.
    Maintains existing signature: takes user_message and optional chat_history, returns string.
    """
    if not OLLAMA_BASE_URL:
        logger.warning("OLLAMA_BASE_URL is not configured. Returning fallback response.")
        import random
        return random.choice(DEMO_RESPONSES)

    # Format messages for Ollama's /api/chat endpoint
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
        }
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            reply = data.get("message", {}).get("content", "").strip()
            if reply:
                return reply
            logger.warning("Empty reply received from Ollama.")
            return FALLBACK_REPLY

    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama endpoint at {url}. Service may be waking up or offline.")
        return FALLBACK_REPLY
    except httpx.TimeoutException:
        logger.error(f"Ollama request timed out after {TIMEOUT_SECONDS} seconds.")
        return FALLBACK_REPLY
    except Exception as e:
        logger.error(f"Unexpected error communicating with Ollama: {e}")
        return FALLBACK_REPLY
