# Local Qwen Chat (Qwen 3.5 Under 7B)

This is a minimal local webapp that lets you pick and chat with the latest open-source Qwen 3.5 models under 7B params.

Included models:
- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-4B-Base`

## 1) Create and activate a Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install deps

```bash
pip install -r requirements.txt
```

## 3) Run app

```bash
python -m uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

## Notes

- On first model load, model files are downloaded from Hugging Face.
- In the UI, select a model, click **Load Model**, then start chatting.
- Shortcut: `Cmd+Enter` (or `Ctrl+Enter`) sends the message.
- CPU works but will be slower than Apple Silicon GPU (`mps`) or CUDA.
