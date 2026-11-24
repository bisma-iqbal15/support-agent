import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: .env support
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    rag_url: str = "http://127.0.0.1:8000"
    max_steps: int = 6
    refund_window_days: int = 14
    orders_path: str = str(ROOT / "data" / "orders.json")
    tickets_path: str = str(ROOT / "data" / "tickets.json")
    refunds_path: str = str(ROOT / "data" / "refunds.json")
    traces_dir: str = str(ROOT / "traces")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", d.ollama_url).rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", d.llm_model),
            rag_url=os.getenv("RAG_URL", d.rag_url).rstrip("/"),
            max_steps=int(os.getenv("MAX_STEPS", d.max_steps)),
            refund_window_days=int(os.getenv("REFUND_WINDOW_DAYS", d.refund_window_days)),
        )
