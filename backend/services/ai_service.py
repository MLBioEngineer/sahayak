"""
Sahayak AI Inference Service.
Connects to Groq API for lightning-fast 24/7 responses using Llama-3.
Implements RAG (Retrieval-Augmented Generation) for accurate medical answers.
"""
import os
import logging
import random
from groq import Groq
from services.rag_service import retrieve_context

logger = logging.getLogger("sahayak.ai_service")

# Configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") # Fast and reliable Groq model
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0"))

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# General Bengali Medical System Prompt
SYSTEM_PROMPT = (
    "আপনি 'সহায়ক' (Sahayak) — একটি সহানুভূতিশীল, পেশাদার এবং নির্ভরযোগ্য বাংলা মেডিকেল এআই সহকারী। "
    "আপনি ব্যবহারকারীর যেকোনো শারীরিক বা স্বাস্থ্য সমস্যা মনোযোগ দিয়ে শুনবেন এবং বাংলায় স্পষ্ট, সহজ ও ব্যবহারোপযোগী পরামর্শ দেবেন। "
    "গুরুত্বপূর্ণ নির্দেশনা:\n"
    "১. সর্বদা খাঁটি ও প্রাঞ্জল বাংলায় উত্তর দিন।\n"
    "২. কখনোই ক্ষতিকর বা অপ্রমাণিত প্রেসক্রিপশন বা ওষুধের নাম সরাসরি দেবেন না।\n"
    "৩. গুরুতর, দীর্ঘস্থায়ী বা তীব্র লক্ষণের ক্ষেত্রে অবিলম্বে নিকটস্থ হাসপাতাল অথবা রেজিস্টার্ড চিকিৎসকের কাছে যাওয়ার স্পষ্ট পরামর্শ দিন।\n"
    "৪. যদি ব্যবহারকারীর প্রশ্নের সাথে সম্পর্কিত কোনো 'মেডিকেল রেফারেন্স (Context)' নিচে দেওয়া থাকে, তবে শুধুমাত্র সেই রেফারেন্সের ওপর ভিত্তি করে উত্তর দিন। নিজে থেকে কোনো বানোয়াট তথ্য (Hallucinate) দেবেন না।"
)

# Resilient Bengali fallback messages when API is not configured
FALLBACK_REPLY = (
    "দুঃখিত, সহায়ক এআই ইঞ্জিন বর্তমানে কনফিগার করা নেই (Groq API Key অনুপস্থিত)। "
    "অনুগ্রহ করে অ্যাডমিনিস্ট্রেটরকে জানান।\n\n"
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
    Generate an AI response using the blazingly fast Groq API.
    Retrieves context from local RAG (FAISS) vector store to ground the response.
    """
    if not client:
        logger.warning("GROQ_API_KEY is not configured. Returning fallback response.")
        return random.choice(DEMO_RESPONSES)
        
    # Retrieve relevant medical context from RAG
    context = retrieve_context(user_message, k=3)
    
    dynamic_system_prompt = SYSTEM_PROMPT
    if context:
        dynamic_system_prompt += f"\n\n--- মেডিকেল রেফারেন্স (Context) ---\n{context}\n-----------------------------------"

    # Format messages for Groq API
    messages = [{"role": "system", "content": dynamic_system_prompt}]

    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
            stream=False,
            timeout=TIMEOUT_SECONDS
        )
        
        reply = chat_completion.choices[0].message.content.strip()
        if reply:
            return reply
            
        logger.warning("Empty reply received from Groq.")
        return FALLBACK_REPLY

    except Exception as e:
        logger.error(f"Unexpected error communicating with Groq API: {e}")
        return "দুঃখিত, এআই সার্ভারের সাথে সংযোগে সমস্যা হচ্ছে। একটু পর আবার চেষ্টা করুন।"
