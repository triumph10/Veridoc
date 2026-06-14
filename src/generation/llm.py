import httpx
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate(prompt: str, model: str = "mistral") -> str:
    if GROQ_API_KEY:
        return _generate_groq(prompt)
    return _generate_ollama(prompt, model)


def _generate_ollama(prompt: str, model: str = "mistral") -> str:
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=300.0
    )
    return response.json()["response"]


def _generate_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content