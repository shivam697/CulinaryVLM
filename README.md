# 🍛 CulinaryVLM

**Multilingual Indian Cooking Video Intelligence, with an agentic GenAI interface**

> A web app where users search cooking videos by technique, ask questions about biryani preparation across regional styles, and see AI-powered procedural comparisons — powered by a VLM fine-tuned on 500+ multilingual Indian cooking videos, exposed through a LangGraph agent that plans, picks tools, and composes answers.

## 🌐 Live Deployment

| Service | URL |
|---|---|
| **Frontend** (React + Vite) | [culinary-vlm.vercel.app](https://culinary-vlm.vercel.app) |
| **Backend API** (FastAPI) | [culinary-vlm-api.onrender.com](https://culinary-vlm-api.onrender.com) |
| **Fine-tuned Model** (HF Hub) | [shivamminde/culinary-vlm-qlora](https://huggingface.co/shivamminde/culinary-vlm-qlora) |
| **Eval Notebook** (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivam697/CulinaryVLM/blob/main/notebooks/evaluate_vlm.ipynb) |

> **Note**: Backend is hosted on Render's free tier — it may take ~30 seconds to wake up on first request.

## 📊 Dataset

- **573 chicken biryani videos** from YouTube, pre-curated and classified
- **21 regional styles**: Hyderabadi, Kolkata, Lucknowi, Muradabadi, Delhi, Andhra, Sindhi, Malabar, Bihari, Assamese, Bombay, Kashmiri, Mughlai, Ambur, Dindigul, Degi, Matka, Tandoori, Bamboo, Arabic, Generic
- **6 languages**: Hindi, Telugu, Malayalam, Bengali, Urdu, English
- **172 videos** selected for the ML pipeline (High + Medium confidence)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  SYSTEM 1: Research Pipeline (Stages 0-10)      │
│  Plain Python • No LangChain • Reproducible     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐  │
│  │ ASR │→│ Seg │→│Align│→│ QA  │→│Fine-tune│  │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────────┘  │
├─────────────────────────────────────────────────┤
│  FastAPI (6 endpoints)                          │
│  /qa  /search  /compare  /recipes  /videos      │
├─────────────────────────────────────────────────┤
│  SYSTEM 2: Agent Layer (OPTIONAL, toggleable)   │
│  LangChain + LangGraph • Feature-flagged        │
│  /agent/query (multi-step reasoning)            │
└─────────────────────────────────────────────────┘
```

System 2 can be deleted entirely and System 1 still works.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/shivam697/CulinaryVLM.git
cd CulinaryVLM

# Setup
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, GEMINI_API_KEY, HF_TOKEN)

# Install
pip install -r requirements.txt
pip install -r requirements-agent.txt  # optional, for agent layer

# Run backend
uvicorn app.main:app --reload --port 8000

# Run frontend
cd frontend && npm install && npm run dev
```

## 📁 Project Structure

```
culinary-vlm/
├── app/                    # FastAPI backend
│   ├── routers/            # API endpoints
│   ├── services/           # Shared business logic
│   ├── agents/             # LangGraph graph (optional)
│   └── tools/              # LangChain tool wrappers (optional)
├── pipeline/               # ML pipeline (Stages 0-10)
├── training/               # Fine-tuning scripts (finetune_qlora.py)
├── frontend/               # React + Vite
├── configs/                # YAML configs + canonical recipes
├── datasets/               # Data artifacts
├── scripts/                # PBS job scripts (GPU cluster)
├── docker/                 # Dockerfiles + nginx
└── tests/                  # Unit + integration tests
```

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Vanilla CSS |
| Backend | FastAPI + FAISS + keyword search |
| ML | Llama-3.2-11B-Vision (QLoRA via Unsloth), WhisperX, InternVL2-8B |
| Agent | LangChain + LangGraph (optional) |
| Deploy | Render (API) + Vercel (frontend) + HF Hub (model) |

---

## 🤗 Fine-Tuned Model on Hugging Face

The fine-tuned **Llama-3.2-11B-Vision** model (QLoRA LoRA adapter, trained on 172 Indian cooking videos) is hosted publicly on Hugging Face Hub:

> **Model**: [`shivamminde/culinary-vlm-qlora`](https://huggingface.co/shivamminde/culinary-vlm-qlora)

The adapter was trained using **Unsloth + QLoRA** (4-bit quantization, LoRA r=16) on a GPU cluster (IIIT Delhi HPC). Running it requires a GPU with **≥14 GB VRAM** — use Google Colab (free T4) for verification.

### Training Details

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` |
| Method | QLoRA (4-bit, LoRA r=16, alpha=32) |
| Vision layers | **Frozen** (`finetune_vision_layers=False`) |
| Language layers | Fine-tuned |
| Inference type | **Text-in → Text-out** (no image needed at inference) |
| Input format | `Category + Context + Question` |

### ⚠️ Hardware Limitation

> Running Llama-3.2-11B-Vision locally requires a dedicated GPU with at least 14 GB VRAM.
> For local development and API testing, the backend uses Groq-hosted LLMs and keyword-based retrieval.
> To **run and verify the fine-tuned VLM**, use **Google Colab** (free T4 GPU) as described below.

---

## 🔬 Verifying the Fine-Tuned Model on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivam697/CulinaryVLM/blob/main/notebooks/evaluate_vlm.ipynb)

Click the badge above to open the full evaluation notebook in one click. It includes:
- Model loading (Steps 1-3)
- Single query inference (Step 4)
- Batch evaluation on the test set with keyword overlap scoring (Step 5)

Or follow the manual steps below:

Since the 11B-parameter model requires a GPU, Google Colab (free T4 tier) is the easiest way to load and test it.

### Step 1 — Open Colab & Set Runtime

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Change runtime type**
3. Select **T4 GPU** → Save

### Step 2 — Install Dependencies

The model was trained with Unsloth — use the same library for inference:

```python
# Install unsloth (same library used for training)
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q huggingface_hub
```

### Step 3 — Authenticate with Hugging Face

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # Get from https://huggingface.co/settings/tokens
```

### Step 4 — Load the Fine-Tuned Adapter

> **Key point**: `shivamminde/culinary-vlm-qlora` is a **LoRA adapter**, not a standalone model.
> Unsloth loads the base model + adapter together automatically.

```python
from unsloth import FastVisionModel
import torch

ADAPTER_ID = "shivamminde/culinary-vlm-qlora"

# Load base model + LoRA adapter (Unsloth handles merging automatically)
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=ADAPTER_ID,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Switch to inference mode
FastVisionModel.for_inference(model)
print("✅ Model loaded!")
```

### Step 5 — Run Text QA Inference

The model takes `Category + Context + Question` as text — matching the training format exactly.
Vision layers were **frozen** during training, so **no image is needed** at inference:

```python
# Format matches training/finetune_qlora.py → format_qa_for_training()
category = "Hyderabadi"
context  = "Action: Adding turmeric powder. Description: The person is adding turmeric powder to the marinated chicken in the bowl."
question = "What ingredients are being used in this step?"

# Build user message — same format as training
user_parts = [f"Category: {category}"]
if context:
    user_parts.append(f"Context: {context}")
user_parts.append(f"Question: {question}")
user_content = "\n\n".join(user_parts)

messages = [{"role": "user", "content": user_content}]

# Apply chat template and tokenize
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate answer
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
    )

# Decode only newly generated tokens (skip the input prompt)
answer = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print("🍛 Model Answer:", answer)
# Expected: "Turmeric powder is being added to the marinated chicken..."
```

### Step 6 — Evaluate on the Test Set (Optional)

Upload `datasets/qa/test_filtered.json` from this repo to your Colab session, then:

```python
import json, torch

# Load test set (upload datasets/qa/test_filtered.json to Colab first)
with open("test_filtered.json") as f:
    test_data = json.load(f).get("qa_pairs", [])

# Filter valid samples (same logic as training)
test_data = [
    qa for qa in test_data
    if not qa.get("needs_generation") and qa.get("question") and qa.get("answer")
]
print(f"Total valid test samples: {len(test_data)}")

# Run inference on first 20 samples
for qa in test_data[:20]:
    category = qa.get("category", "biryani")
    context  = qa.get("context", "")
    question = qa.get("question", "")
    expected = qa.get("answer", "")

    parts = [f"Category: {category}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {question}")

    messages = [{"role": "user", "content": "\n\n".join(parts)}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, use_cache=True
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"\nQ: {question[:80]}")
    print(f"Expected : {expected[:100]}")
    print(f"Generated: {answer[:100]}")
```

---

## 📜 Based On

Extension of ICVGIP 2025 paper *"How Does India Cook Biryani?"*

## 👤 Author

**Shivam** — M.Tech CSE, IIIT Delhi (2025-2027)
Research Intern, CosyLab (Dr. Ganesh Bagler)
