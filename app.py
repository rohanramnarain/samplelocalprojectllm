from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_repo: str
    label: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen3.5-0.8b": ModelSpec(
        id="qwen3.5-0.8b",
        hf_repo="Qwen/Qwen3.5-0.8B",
        label="Qwen3.5 0.8B",
    ),
    "qwen3.5-0.8b-base": ModelSpec(
        id="qwen3.5-0.8b-base",
        hf_repo="Qwen/Qwen3.5-0.8B-Base",
        label="Qwen3.5 0.8B Base",
    ),
    "qwen3.5-2b": ModelSpec(
        id="qwen3.5-2b",
        hf_repo="Qwen/Qwen3.5-2B",
        label="Qwen3.5 2B",
    ),
    "qwen3.5-2b-base": ModelSpec(
        id="qwen3.5-2b-base",
        hf_repo="Qwen/Qwen3.5-2B-Base",
        label="Qwen3.5 2B Base",
    ),
    "qwen3.5-4b": ModelSpec(
        id="qwen3.5-4b",
        hf_repo="Qwen/Qwen3.5-4B",
        label="Qwen3.5 4B",
    ),
    "qwen3.5-4b-base": ModelSpec(
        id="qwen3.5-4b-base",
        hf_repo="Qwen/Qwen3.5-4B-Base",
        label="Qwen3.5 4B Base",
    ),
}


class SelectModelRequest(BaseModel):
    model_id: str = Field(..., description="Model id from /api/models")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatTurn] = Field(default_factory=list)
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class LocalQwenEngine:
    def __init__(self) -> None:
        self.current_model_id: Optional[str] = None
        self.tokenizer = None
        self.model = None
        self.device = self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _dtype_for_device(self) -> torch.dtype:
        if self.device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    def unload(self) -> None:
        self.tokenizer = None
        self.model = None
        self.current_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_id: str) -> Dict[str, Any]:
        spec = MODEL_SPECS.get(model_id)
        if not spec:
            raise ValueError(f"Unknown model_id: {model_id}")

        if self.current_model_id == model_id and self.model is not None and self.tokenizer is not None:
            return {
                "model_id": model_id,
                "label": spec.label,
                "hf_repo": spec.hf_repo,
                "device": self.device,
                "already_loaded": True,
            }

        self.unload()

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo,
            trust_remote_code=True,
            torch_dtype=self._dtype_for_device(),
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.current_model_id = model_id

        return {
            "model_id": model_id,
            "label": spec.label,
            "hf_repo": spec.hf_repo,
            "device": self.device,
            "already_loaded": False,
        }

    def generate(
        self,
        message: str,
        history: List[ChatTurn],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        if self.model is None or self.tokenizer is None or self.current_model_id is None:
            raise RuntimeError("No model loaded. Call /api/select-model first.")

        messages = []
        for turn in history:
            if turn.role not in {"user", "assistant", "system"}:
                continue
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return reply.strip()


app = FastAPI(title="Local Qwen Chat")
engine = LocalQwenEngine()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    return {
        "device": engine.device,
        "current_model_id": engine.current_model_id,
        "models": [
            {
                "id": spec.id,
                "label": spec.label,
                "hf_repo": spec.hf_repo,
                "loaded": spec.id == engine.current_model_id,
            }
            for spec in MODEL_SPECS.values()
        ],
    }


@app.post("/api/select-model")
def select_model(payload: SelectModelRequest) -> Dict[str, Any]:
    try:
        return engine.load(payload.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    try:
        reply = engine.generate(
            message=payload.message,
            history=payload.history,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        return {
            "reply": reply,
            "model_id": engine.current_model_id,
            "device": engine.device,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
