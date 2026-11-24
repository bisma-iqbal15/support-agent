"""LLM client. The agent only needs a function: (messages, json_mode) -> text."""


class LLMError(RuntimeError):
    pass


def make_ollama_llm(settings):
    def chat(messages: list[dict], json_mode: bool = True) -> str:
        import httpx

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        if json_mode:
            payload["format"] = "json"  # Ollama constrains the output to valid JSON
        try:
            r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=300)
        except httpx.ConnectError:
            raise LLMError("Can't reach Ollama. Open the Ollama app and try again.")
        except httpx.TimeoutException:
            raise LLMError("The model took too long to respond.")
        if r.status_code == 404:
            raise LLMError(f"Model not found. Run: ollama pull {settings.llm_model}")
        r.raise_for_status()
        return r.json()["message"]["content"]

    return chat
