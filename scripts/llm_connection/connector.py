import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT_PATH = Path(__file__).with_name("SYSTEM_PROMPT.txt")


def read_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def get_api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Render environment variables "
            "or Streamlit secrets before using the chat tab."
        )
    return key


def ask_llm(
    user_query: str,
    original_summary: dict,
    filtered_summary: dict,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    from groq import Groq

    client = Groq(api_key=get_api_key(api_key))
    system_prompt = read_system_prompt()
    payload = {
        "user_query": user_query,
        "original_dataset_summary": original_summary,
        "current_filtered_dataset_summary": filtered_summary,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, default=str, ensure_ascii=False),
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
