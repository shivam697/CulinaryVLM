# 🍛 CulinaryVLM

**Multilingual Indian Cooking Video Intelligence, with an agentic GenAI interface**

> A web app where users search cooking videos by technique, ask questions about biryani preparation across regional styles, and see AI-powered procedural comparisons — powered by a VLM fine-tuned on 500+ multilingual Indian cooking videos, exposed through a LangGraph agent that plans, picks tools, and composes answers.

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
│  FastAPI (6 original endpoints)                 │
│  /qa  /search  /compare  /recipes  /videos      │
├─────────────────────────────────────────────────┤
│  SYSTEM 2: Agent Layer (OPTIONAL, toggleable)   │
│  LangChain + LangGraph • Feature-flagged        │
│  /agent/query (8 tools, multi-step reasoning)   │
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

# Run pipeline (Phase 0)
python pipeline/00a_categorize.py
python pipeline/00b_canonical_recipes.py
python pipeline/00c_metadata_tagging.py

# Docker (full stack)
docker-compose up --build
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
├── training/               # Fine-tuning scripts
├── frontend/               # React + Vite + TailwindCSS
├── configs/                # YAML configs + canonical recipes
├── datasets/               # Data artifacts
├── scripts/                # PBS job scripts (GPU cluster)
├── docker/                 # Dockerfiles + nginx
└── tests/                  # Unit + integration tests
```

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + TailwindCSS |
| Backend | FastAPI + SQLite + FAISS |
| ML | Llama-3.2-11B-Vision (QLoRA), WhisperX, InternVL2-8B |
| Agent | LangChain + LangGraph (optional) |
| Deploy | Render (API) + GitHub Pages (frontend) + HF Hub (model) |

---

## 🤗 Fine-Tuned Model on Hugging Face

The fine-tuned **Llama-3.2-11B-Vision** model (QLoRA, trained on 172 Indian cooking videos) is hosted publicly on Hugging Face Hub:

> **Model**: [`shivamminde/culinary-vlm`](https://huggingface.co/shivamminde/culinary-vlm)

The model was fine-tuned using QLoRA (4-bit quantization) on a GPU cluster (IIIT Delhi HPC). Running it locally requires **~14 GB+ VRAM** (an A100/V100 GPU), which is not feasible on most personal machines (e.g., MacBooks with Apple Silicon).

### ⚠️ Hardware Limitation

> Running Llama-3.2-11B-Vision locally requires a dedicated GPU with at least 14 GB VRAM.  
> For local development and API testing, the backend falls back to Groq-hosted LLMs and FAISS-based retrieval.  
> To actually **run and verify the fine-tuned VLM**, use **Google Colab** (free T4 GPU) as described below.

---

## 🔬 Verifying the Fine-Tuned Model on Google Colab

Since the 11B-parameter model requires a GPU, Google Colab (free T4 tier) is the easiest way to load and test it.

### Step 1 — Open Colab & Set Runtime

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Change runtime type**
3. Select **T4 GPU** → Save

### Step 2 — Install Dependencies

```python
!pip install -q transformers accelerate bitsandbytes pillow huggingface_hub
```

### Step 3 — Authenticate with Hugging Face

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # Get from https://huggingface.co/settings/tokens
```

### Step 4 — Load the Fine-Tuned Model

> **Note**: The model was fine-tuned on **text-only QA pairs** (transcript context + question → answer). The vision layers were **frozen** during training (`finetune_vision_layers=False`). This is a **text-in, text-out** QA model — no images needed at inference.

```python
from transformers import AutoTokenizer, AutoModelForVision2Seq, BitsAndBytesConfig
import torch

MODEL_ID = "shivamminde/culinary-vlm"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
print("✅ Model loaded!")
```

### Step 5 — Run Text QA Inference

The model takes `Category + Context + Question` as text input — matching the training format exactly:

```python
# Format matches datasets/qa/train_filtered.json
category = "Hyderabadi"
context  = "Action: Adding turmeric powder. Description: The person is adding turmeric powder to the marinated chicken in the bowl."
question = "What ingredients are being used in this step?"

user_content = f"Category: {category}\n\nContext: {context}\n\nQuestion: {question}"
messages = [{"role": "user", "content": user_content}]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

# Decode only the newly generated tokens
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("🍛 Model Answer:", answer)
# Expected: "Turmeric powder and marinated chicken are being used..."
```

### Step 6 — Evaluate on the Test Set (Optional)

Upload `datasets/qa/test_filtered.json` from this repo to your Colab session, then:

```python
import json, torch

# Load test set (upload datasets/qa/test_filtered.json to Colab first)
with open("test_filtered.json") as f:
    test_data = json.load(f).get("qa_pairs", [])

# Filter valid samples
test_data = [
    qa for qa in test_data
    if not qa.get("needs_generation") and qa.get("question") and qa.get("answer")
]
print(f"Total test samples: {len(test_data)}")

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
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {question[:80]}")
    print(f"Expected: {expected[:100]}")
    print(f"Got:      {answer[:100]}")
```

> **Tip**: For a full evaluation notebook, see [`notebooks/evaluate_vlm.ipynb`](notebooks/evaluate_vlm.ipynb).

---

## 📜 Based On

Extension of ICVGIP 2025 paper *"How Does India Cook Biryani?"*

## 👤 Author

**Shivam** — M.Tech CSE, IIIT Delhi (2025-2027)
Research Intern, CosyLab (Dr. Ganesh Bagler)
