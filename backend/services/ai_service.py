"""
This is the ONE function you'll replace once your fine-tuned model is ready.

Right now it returns a canned/mock response so the rest of the site
(frontend, auth, PDF export) can be built and tested without a real model.

Later, swap the body of get_ai_response() to call your self-hosted model,
e.g. a request to your HF Space / Colab-hosted inference endpoint, or a
local transformers pipeline. Nothing else in the codebase needs to change
as long as this function still takes a string and returns a string.
"""

MOCK_RESPONSES = [
    "এটা সাধারণ ভাইরাল উপসর্গের মতো শোনাচ্ছে। পর্যাপ্ত পানি ও বিশ্রাম নিন এবং লক্ষণ ৩ দিনের বেশি থাকলে ডাক্তার দেখান।",
    "আপনার লক্ষণগুলো নিয়ে আরও তথ্য দরকার — জ্বর কতদিন ধরে আছে এবং তাপমাত্রা কত?",
    "এটি একটি ডেমো রেসপন্স। আসল মডেল বসানো হলে এখানে RAG-grounded উত্তর আসবে।",
]


def get_ai_response(user_message: str, chat_history: list[dict] | None = None) -> str:
    """
    Mock implementation.

    Args:
        user_message: the latest message from the user
        chat_history: list of {"role": "user"|"assistant", "content": str}

    Returns:
        The assistant's reply as plain text.
    """
    # --- MOCK LOGIC (delete this block when wiring up the real model) ---
    import random
    return random.choice(MOCK_RESPONSES)
    # ----------------------------------------------------------------
    #
    # --- REAL MODEL WIRING (uncomment / adapt when ready) ---
    # response = requests.post(
    #     "https://your-model-endpoint/generate",
    #     json={"prompt": user_message, "history": chat_history},
    # )
    # return response.json()["text"]
