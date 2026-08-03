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
git clone https://github.com/{username}/culinary-vlm.git
cd culinary-vlm

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

## 📜 Based On

Extension of ICVGIP 2025 paper *"How Does India Cook Biryani?"*

## 👤 Author

**Shivam** — M.Tech CSE, IIIT Delhi (2025-2027)
Research Intern, CosyLab (Dr. Ganesh Bagler)
