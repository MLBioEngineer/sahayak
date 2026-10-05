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
    "আপনি 'সহায়ক' (Sahayak) — একটি সহানুভূতিশীল, পেশাদার এবং নির্ভরযোগ্য বাংলা মেডিকেল এআই সহকারী ও স্লিপ অ্যাপনিয়া বিশ্লেষক। "
    "ব্যবহারকারীর সাধারণ স্বাস্থ্য সমস্যা ও স্লিপ অ্যাপনিয়া (Sleep Apnea / OSA) সংক্রান্ত ইসিজি ও লক্ষণগুলো মনোযোগ দিয়ে শুনুন এবং বাংলায় স্পষ্ট, সহজ ও ব্যবহারোপযোগী প্রাথমিক পরামর্শ দিন। "
    "স্লিপ অ্যাপনিয়া সংক্রান্ত ক্লিনিক্যাল জ্ঞান:\n"
    "- AHI (Apnea-Hypopnea Index): <৫ স্বাভাবিক, ৫-১৪.৯ মৃদু, ১৫-২৯.৯ মাঝারি, ≥৩০ মারাত্মক স্লিপ অ্যাপনিয়া।\n"
    "- সাধারণ উপসর্গ: তীব্র নাক ডাকা, ঘুমে দম বন্ধ লাগা, দিনের বেলা অতিরিক্ত ক্লান্তি, সকালে মাথাব্যথা।\n"
    "- প্রাথমিক জীবনযাত্রা পরামর্শ: চিৎ হয়ে না ঘুমিয়ে পাশ ফিরে (Lateral position) ঘুমানো, ওজন কমানো, ধূমপান পরিহার।\n"
    "- নিশ্চিত ডায়াগনসিসের জন্য পলিসমনোগ্রাফি (PSG) টেস্ট এবং প্রয়োজনে CPAP থেরাপির পরামর্শ দিন।\n"
    "গুরুত্বপূর্ণ নির্দেশনা:\n"
    "১. সর্বদা খাঁটি ও প্রাঞ্জল বাংলায় উত্তর দিন।\n"
    "২. কখনোই ক্ষতিকর বা অপ্রমাণিত প্রেসক্রিপশন দেবেন না।\n"
    "৩. গুরুতর, দীর্ঘস্থায়ী বা তীব্র লক্ষণের ক্ষেত্রে অবিলম্বে নিকটস্থ হাসপাতাল অথবা রেজিস্টার্ড চিকিৎসকের কাছে যাওয়ার স্পষ্ট পরামর্শ দিন।"
)

# Resilient Bengali fallback messages when LLM is cold-starting or temporarily unreachable
FALLBACK_REPLY = (
    "দুঃখিত, সহায়ক এআই ইঞ্জিন বর্তমানে কিছুটা ব্যস্ত রয়েছে অথবা মডেলটি চালু হচ্ছে। "
    "অনুগ্রহ করে এক মিনিট পর আবার বার্তা পাঠান।\n\n"
    "জরুরি শারীরিক সমস্যা বা মারাত্মক উপসর্গের ক্ষেত্রে অনুগ্রহ করে বিলম্ব না করে নিকটস্থ হাসপাতালে যান "
    "অথবা জাতীয় জরুরি সেবা ৯৯৯ এ যোগাযোগ করুন।"
)

DEMO_RESPONSES = [
    "স্লিপ অ্যাপনিয়া (Sleep Apnea) হলো ঘুমের মধ্যে শ্বাসনালী সাময়িকভাবে সংকুচিত হয়ে শ্বাস বন্ধ হয়ে যাওয়ার একটি সমস্যা। এর প্রধান লক্ষণ হলো উচ্চ শব্দে নাক ডাকা এবং সকালে ক্লান্ত বোধ করা। এর প্রতিকারে পাশ ফিরে ঘুমানো এবং ওজন নিয়ন্ত্রণ কার্যকর ভূমিকা রাখে।",
    "আপনার ইসিজি ও স্লিপ অ্যাপনিয়া ট্র্যাকিং ফলাফল পর্যালোচনা করা হয়েছে। AHI স্কোর যদি ৫-এর বেশি হয়, তবে এটি শ্বাসরোধের লক্ষণ নির্দেশ করে। সঠিক রোগ নির্ণয়ের জন্য একজন বক্ষব্যাধি বিশেষজ্ঞের শরণাপন্ন হয়ে পলিসমনোগ্রাফি (PSG) টেস্ট করানো প্রয়োজন।",
    "এটি সাধারণ স্বাস্থ্য সুরক্ষার প্রাথমিক পরামর্শ: পর্যাপ্ত পানি পান করুন, পুষ্টিকর খাবার গ্রহণ করুন এবং বিশ্রাম নিন। লক্ষণ ৩ দিনের বেশি স্থায়ী হলে বা তীব্র আকার ধারণ করলে নিকটস্থ চিকিৎসকের পরামর্শ নিন।"
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
