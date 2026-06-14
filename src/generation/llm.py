import httpx

def generate(prompt :  str,  model: str = "mistral") -> str:
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt" : prompt,
            "stream": False
        },
        timeout = 300.0
    )
    
    
    return response.json()["response"]
