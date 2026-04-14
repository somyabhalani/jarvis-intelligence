<img width="1359" height="1062" alt="Screenshot 2026-04-15 022338" src="https://github.com/user-attachments/assets/c25e8cc7-057d-4cf0-a53f-85d2970b21dc" />
<div align="center">
![WhatsApp Image 2026-04-15 at 2 24 44 AM](https://github.com/user-attachments/assets/27714f74-5133-430b-b606-fdaaacc4ef98)


# ⚡ JARVIS AI

### *Just A Rather Very Intelligent System*

**A personal Iron Man-inspired AI assistant with multi-model intelligence, live vision, voice control, and a real-time HUD dashboard.**

---

> **Architect of Jarvis Intelligence:** [Somya Bhalani](https://github.com/somyabhalani)

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM%20API-76b900?style=flat-square&logo=nvidia)](https://integrate.api.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#)

</div>

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Architecture — How the Brain Thinks](#architecture--how-the-brain-thinks)
3. [Model Pipeline](#model-pipeline)
4. [All Features](#all-features)
5. [Commands & Expected Outputs](#commands--expected-outputs)
6. [How Everything Works (Deep Dive)](#how-everything-works-deep-dive)
7. [Project Structure](#project-structure)
8. [How to Run](#how-to-run)
9. [Configuration](#configuration)
10. [Tech Stack](#tech-stack)

---

## Overview

Jarvis is a fully local, self-hosted AI assistant inspired by Tony Stark's FRIDAY/JARVIS. It runs in your browser as a dark HUD dashboard and communicates with a Python backend that orchestrates multiple large language models, live vision, real-time web data, and system control — all through a single unified brain.

**What makes it different from ChatGPT:**
- It **knows your local context** — your current city, weather, traffic, news — without you asking
- It **routes intelligently** — a coding question gets the coder model; a plan gets the planning model; casual chat gets super-fast response
- It **watches** — the camera pipeline detects your mood, objects in frame, faces, and proactively speaks when something changes
- It **learns** — formality, humor, depth, and verbosity adapt from every interaction
- It **controls** — open apps, websites, YouTube; issue browser commands by voice

---

## Architecture — How the Brain Thinks

```
User Input (Text / Voice / Image)
           │
           ▼
┌──────────────────────────────────────┐
│           IntentRouter               │
│  (heuristic regex-based routing)     │
│  → system / research / analysis /    │
│    planning / memory / conversation  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│       IntentClassifierLLM            │
│  (meta/llama-3.1-8b-instruct)        │
│  Verifies or overrides heuristic     │
│  intent with confidence score        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│            Brain._plan()             │
│  Routes to the correct handler:      │
│  research → ResearchService          │
│  system   → SystemController         │
│  planning → planning model           │
│  memory   → MemoryStore              │
│  convo    → DialogueTuner + LLM      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         LLMOrchestrator              │
│  _route_task_fast() picks best lane  │
│  then tries candidate models in      │
│  priority order with fallback chain  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│       PersonaEngine.style_response() │
│  Adds Jarvis tone, opener, phrasing  │
│  based on user mood signal          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│        _avoid_repetition()           │
│  Detects identical responses and     │
│  appends reframe suggestion          │
└──────────────┬───────────────────────┘
               │
               ▼
         Final Response
     (stored in MemoryStore)
```

### How the Brain Thinks — Step by Step

1. **Signal Analysis** — `UserModel.infer_style()` reads the raw text and extracts: urgency, stress, formality, humor preference, task intensity, mood (focused/stressed/neutral), and text style (concise/balanced/detailed).

2. **Heuristic Routing** — `IntentRouter.route()` uses fast regex patterns to assign an initial intent label and confidence score without any API call. High-confidence system queries (local time, weather, GPU, IP, traffic) are locked at ≥0.94 confidence and skip the classifier entirely.

3. **LLM Intent Classification** — The `IntentClassifierLLM` (llama-3.1-8b) only runs when the heuristic confidence is below the lock threshold. It validates or overrides the intent category.

4. **Plan Execution** — `Brain._plan()` dispatches to specialized handlers per intent. Each handler knows exactly which models to call and what context to provide.

5. **Task-based Model Routing** — `LLMOrchestrator._route_task_fast()` applies another layer of smart routing: it pattern-matches the user query for code, planning, research, etc., and picks a task lane. The `_heuristic_route_task()` runs locally first for zero-latency routing on obvious patterns.

6. **Response Generation** — The orchestrator tries candidate models in priority order (primary → secondary → fallback chain → local rule-based). If NVIDIA NIM is unreachable, `RuleBasedLLMClient` synthesizes a useful response locally.

7. **Persona Styling** — `PersonaEngine.style_response()` prepends personality openers (Certainly / Right away / Absolutely) based on user urgency and formality profile. Short replies (<180 chars) are left unstyled.

8. **Memory Persistence** — The full response + intent + metadata is committed to SQLite for long-term memory and future recall.

---

## Model Pipeline

The system uses **NVIDIA NIM API** which hosts multiple open-weight models. Each task lane uses the best-fit model:

| Lane | Model | When Used |
|------|-------|-----------|
| **Default / General** | `deepseek-ai/deepseek-v3.1-terminus` | All-purpose queries |
| **Fast / Conversation** | `meta/llama-3.1-8b-instruct` | Quick replies, greetings, small talk |
| **Intent Router** | `meta/llama-3.1-8b-instruct` | Classifying user intent |
| **Planning** | `meta/llama-3.1-70b-instruct` | Strategy, roadmaps, phase-by-phase plans |
| **Coding** | `deepseek-ai/deepseek-v3.1-terminus` | Code, debug, refactor |
| **Reasoning** | `nvidia/llama-3.3-nemotron-super-49b-v1` | Analysis, deep reasoning, complex logic |
| **Long Context / Deep Research** | `meta/llama-3.1-405b-instruct` | Research with web sources, long documents |
| **Vision (Image Analysis)** | `meta/llama-3.2-11b-vision-instruct` | Describing camera frames, uploaded images |

### Model Selection Logic

```
User asks "write a Python function to sort a list"
       │
       ▼
heuristic: matches /code|python|bug|debug|refactor/ → task = "coding"
       │
       ▼
candidate_models("coding") = [deepseek-v3.1, llama-3.1-405b, nemotron-49b, llama-3.1-8b]
       │
       ▼
Try deepseek-v3.1 → response received → DONE
```

```
User asks "summarize what I said earlier"
       │
       ▼
intent = "summary" (heuristic regex match)
       │
       ▼
Pulls last 8 conversation turns from SQLite memory
Sends to: complete_for_task("summary") → reasoning model
       │
       ▼
Returns clean bullet-point summary
```

### Generation Profiles (Temperature / Tokens per Lane)

| Task | Temperature | Max Tokens |
|------|-------------|-----------|
| `conversation_fast` | 0.45 | 220 |
| `conversation` | 0.50 | 520 |
| `planning` | 0.25 | 900 |
| `coding` | 0.20 | 900 |
| `summary` | 0.20 | 420 |
| `research` | 0.22 | 820 |
| `research_deep` | 0.15 | 1200 |
| `analysis` | 0.20 | 900 |

---

## All Features

### 🧠 Intelligence & Reasoning
- **Multi-model pipeline** — 7 specialized models, auto-selected by task type
- **Reasoning modes:** `fast mode`, `balanced mode`, `deep mode`
- **Intent classification** — 2-stage: heuristic + LLM verifier
- **Research engine** — DuckDuckGo web search + Wikipedia + page scraping + LLM synthesis
- **Planning mode** — Phase-by-phase strategic plans with risks and next steps
- **Summarization** — Summarize any text or the last N conversation turns
- **Repetition avoidance** — Detects repeated replies and offers reframing

### 🎤 Voice & Language
- **Wake phrase:** Say `Hey Jarvis` to activate hands-free
- **Hotkey:** `Ctrl+Space` to trigger voice capture
- **Web Speech API** — In-browser voice recognition (no server-side audio needed)
- **TTS (Text-to-Speech)** — Browser-native speech synthesis with toggle
- **Multilingual:** `English`, `Hinglish`, `Hindi` modes — auto-detect or manually set
- **Voice orb** — Animated particle field visualizes mic energy in real-time

### 👁️ Vision System (Camera + AI)
- **Real-time camera feed** — Browser webcam, no server-side camera needed
- **20-second periodic scan** — Captures frame every 20s, sends to NVIDIA vision model
- **Object detection** — TensorFlow.js COCO-SSD model runs in-browser (80 object classes)
- **Face detection + expression recognition** — face-api.js (tiny face detector + expression net)
- **MobileNet scene classification** — Frame-level scene labeling runs in-browser
- **BLIP captioning** — Xenova/transformers runs BLIP image captioning in WASM
- **Backend vision analysis** — Full frame sent to `meta/llama-3.2-11b-vision-instruct` for rich descriptions
- **Mood tracking** — Brightness-based heuristic + face expression analysis with history
- **Visual timeline** — Logs last 10 detection events with timestamp, mood, object list
- **Auto-announcements** — Proactively speaks if mood/objects change significantly when you're idle >12s

### 🌍 Live Data
- **Weather** — wttr.in free API, auto-detects your location via IP
- **Gold & Silver prices** — metals.live + exchangerate-api (USD → INR conversion)
- **City News** — GDELT live news stream filtered by your detected city
- **Global News** — RSS feeds from BBC, Reuters, Google News, Hacker News
- **City Traffic** — Google Maps traffic links for city or route queries
- **IP-based location** — Multiple geolocation fallbacks (ipwho.is, ip-api, ipinfo, ipapi, geodb)

### 💻 System Control
- **Open apps:** Chrome, Edge, Notepad, Calculator, CMD
- **Open websites:** Google, YouTube, Gmail, Instagram, Facebook, X/Twitter, LinkedIn, Reddit, WhatsApp, Telegram
- **Open any URL** by voice or text
- **YouTube search + autoplay** — Finds first result by videoId and opens with autoplay
- **Browser controls:** pause/play, mute/unmute, fullscreen, volume, scroll up/down, next/prev tab, new tab
- **Clipboard:** read and write clipboard content
- **System stats:** CPU, RAM, disk usage via psutil
- **GPU report:** nvidia-smi / wmic detection
- **IP report:** local + public IP detection
- **Time/Date/Day:** local system time, current date, day of week

### 🧬 Memory & Learning
- **SQLite persistence** — All conversations stored at `~/.jarvis/jarvis.db`
- **MemoryStore tables:** turns, preferences, facts, learning_events, reminders, mood_samples, semantic_memory
- **Semantic search** — Token-overlap search across stored memory
- **User model learning** — formality, humor, technical depth, stress sensitivity adapt from each turn
- **Feedback loop** — Positive/negative feedback adjusts humor, verbosity, proactivity scores
- **Preference storage** — Save/recall/forget specific user facts
- **1-hour answer cache** — Repeated research queries served from cache

### 🖥️ HUD Dashboard (Web UI)
- **Live clock** — Real-time HH:MM:SS + day + date
- **Animated core rings** — Pulsing orb that reflects Jarvis state
- **Animated EQ bars** — Response to voice energy
- **System resource panel** — CPU, RAM, disk gauges
- **Weather widget** — Live temp, condition, forecast
- **City & Global News panels** — Auto-refresh headlines
- **Market data panel** — Crypto and stock tickers
- **Command log** — Full conversation history with intent labels and timestamps
- **Visual detection panel** — Mood display, detected objects, visual timeline, camera feed
- **Learning telemetry** — Reasoning mode, intent, confidence, user profile values

---

## Commands & Expected Outputs

### 💬 Conversation

| Command | Output |
|---------|--------|
| `hi` / `hello` / `hey` | Greeting with mode options and local updates offer |
| `how are you` | Status reply with task suggestions |
| `thank you` | Acknowledgment with offer to continue |
| `what can you do` | Full capabilities overview |
| `good night` / `bye` | Farewell message |
| `roast me` | Light roast with humor |
| `joke` | Programmer joke from rotating pool |
| `dad joke` | Classic dad joke |
| `pun` | Wordplay pun |
| `one liner` | Sharp one-liner |
| `motivate me` | Motivational momentum starter |

### ⚙️ Mode Commands

| Command | Output |
|---------|--------|
| `fast mode` | Enables fast reasoning: low-latency short replies |
| `deep mode` | Enables deep reasoning: rich, detailed responses |
| `balanced mode` | Default: speed and depth in equilibrium |
| `talkative mode peak` | Longer, richer casual replies |
| `talkative mode normal` | Moderate verbosity |
| `talkative mode off` | Concise mode: tight replies |
| `hinglish mode` | Responds in Hinglish |
| `hindi mode` | Responds in Hindi (Devanagari) |
| `english mode` | Responds in English |
| `auto language mode` | Auto-detects Hindi/Hinglish/English per message |

### 🔍 Research & Analysis

| Command | Output |
|---------|--------|
| `what is quantum computing` | Fast answer from LLM |
| `research quantum computing` | DuckDuckGo search + page scraping + Wikipedia + LLM synthesis |
| `deep research quantum computing` | Extended research with long-context model, more sources |
| `research quantum computing with sources` | Full research report with source titles and URLs |
| `analyze the difference between REST and GraphQL` | Detailed comparison via reasoning model |
| `summarize` | Summarizes last 8 conversation turns |
| `summarize [text]` | Summarizes the provided text |

### 🌦️ Live Data

| Command | Output |
|---------|--------|
| `weather` | Current weather for auto-detected location (temp, humidity, wind, forecast) |
| `weather in Mumbai` | Weather for specified city |
| `gold price` | Live 24-carat gold price in INR |
| `silver price` | Live silver price per kg in INR |
| `local news` | City news headlines from GDELT for detected city |
| `city news` | Same as local news |
| `global news` | World headlines from BBC/Reuters/Google News |
| `tech news` | Technology category global news |
| `traffic` | Google Maps traffic for your city |
| `traffic in Delhi` | Traffic for specified city |
| `traffic from Bandra to Andheri` | Route-specific traffic link |

### 💻 System & Apps

| Command | Output |
|---------|--------|
| `open chrome` | Launches Google Chrome |
| `open notepad` | Launches Notepad |
| `open calculator` | Launches Calculator |
| `open youtube` | Opens youtube.com in browser |
| `open gmail` | Opens mail.google.com |
| `open instagram` | Opens instagram.com |
| `open whatsapp` | Opens web.whatsapp.com |
| `open https://example.com` | Opens the specified URL |
| `what time is it` | Current system time (e.g., `04:30 PM`) |
| `what is today's date` | Full date (e.g., `Tuesday, 14 April 2026`) |
| `what day is it` | Day name (e.g., `Tuesday`) |
| `which year` | `Current year is 2026.` |
| `my ip address` | Local + public IP report |
| `gpu usage` | NVIDIA GPU VRAM, utilization, temperature |
| `system stats` | CPU%, RAM%, Disk usage JSON |
| `clipboard read` | Returns current clipboard contents |
| `clipboard write [text]` | Writes text to clipboard |
| `show model pipeline` | Full model routing overview |
| `which model are you using` | Lists all pipeline models by lane |

### 🎬 YouTube & Browser

| Command | Output |
|---------|--------|
| `play [song/video] on youtube` | Searches YouTube, opens first result with autoplay |
| `pause` | Sends spacebar key to pause/resume |
| `mute` / `unmute` | Toggles mute (M key) |
| `fullscreen` | Toggles fullscreen (F key) |
| `volume up` / `volume down` | Arrow key volume steps |
| `set volume 60` | Jumps to ~60% volume |
| `scroll down` / `scroll up` | Page down/up |
| `next tab` | Ctrl+Tab |
| `new tab` | Ctrl+T |
| `close tab` | Ctrl+W |

### 👁️ Vision

| Command | Output |
|---------|--------|
| `what's in my hand` | Captures frame, sends to vision model, describes held object |
| `what am I holding` | Same as above |
| `what is this` | Visual object identification from frame |
| `describe this image` + attach image | Describes uploaded image with vision model |
| `camera on` | Activates webcam feed |
| `camera off` | Stops webcam |
| `mood summary` | Summary of mood history from last N scans |
| `what's going on` | Visual scene summary (mood, objects, people count, scan age) |

### 🧠 Memory

| Command | Output |
|---------|--------|
| `remember [fact]` | Stores fact in preferences DB |
| `forget [key]` | Removes stored preference |
| `what do you know` | Describes memory capability |

---

## How Everything Works (Deep Dive)

### Request Flow — Web Interface

```
Browser (index.html + app.js)
       │
       │  POST /api/command  { text, location_hint, image_data_url }
       ▼
JarvisHandler (ThreadingHTTPServer — server.py)
       │
       ├─ Vision check? → calls vision_check() → NVIDIA API
       ├─ Weather? → JarvisRuntime.system_weather() → wttr.in
       ├─ Traffic? → SystemController.get_city_traffic_report()
       ├─ News? → city_news() → GDELT API
       ├─ GPU/IP? → get_gpu_report() / get_ip_report()
       └─ Everything else → JarvisRuntime.execute() → Brain.ingest()
                                                  │
                                                  ▼
                                          LLMOrchestrator
                                          → NVIDIA NIM API
                                          ← Response text
       │
       │  JSON response { ok, intent, confidence, response, metadata }
       ▼
Browser renders response + speaks via Web Speech API
```

### Vision Pipeline (Frontend)

```
Camera stream (320×240) → canvas frame capture (every 20s)
       │
       ├─ [In-browser] COCO-SSD → object boxes + class labels
       ├─ [In-browser] face-api.js → face count + dominant expression
       ├─ [In-browser] MobileNet → scene classification
       ├─ [In-browser] BLIP (WASM) → image caption text
       └─ [Backend] POST /api/vision-check → llama-3.2-11b-vision → rich description
                │
                ▼
        detectionState updated
        → sent to /api/detection (backend context injection)
        → may trigger auto-speech if mood/objects changed
```

### Research Pipeline

```
User: "research [topic]"
       │
       ▼
ResearchService.answer()
       │
       ├─ search_web() → DuckDuckGo HTML scrape → top 5 URLs
       ├─ fetch_page_text() → raw text from each URL (3 max)
       ├─ wikipedia_lookup() → Wikipedia REST API summary
       └─ summarize() → sends source chunks + wiki to:
              LLMOrchestrator.complete_for_task("research_deep")
              → llama-3.1-405b-instruct (long context)
              → Full grounded answer
              
       If model returns generic reply → _heuristic_report() fallback
       (structured report with wiki, excerpts, themes, source list)
```

### Memory System (SQLite)

```
~/.jarvis/jarvis.db
├─ turns          — Full conversation history with role, content, timestamp, metadata
├─ preferences    — User style preferences (formality, humor, verbosity, etc.)
├─ facts          — Key-value facts by category (user_profile, interests, etc.)
├─ learning_events— Feedback ratings and interaction events
├─ reminders      — Reminder items with due dates and status
├─ mood_samples   — Historical mood + stress + focus readings
└─ semantic_memory— Searchable semantic entries with token-overlap scoring
```

### User Learning Adaptation

Every interaction updates the user profile:
- Long messages (≥20 tokens) → `preferred_length = "detailed"`
- Stress markers → `stress_sensitivity` increases, `humor` decreases
- Formal phrasing → `formality` score increases
- Research/analysis intent → `technical_depth` increases
- Positive feedback (≥0.8) → `humor` and `proactive` increase
- Negative feedback (≤0.4) → `humor` and `verbosity` decrease

---

## Project Structure

```
Documents/
│
├── jarvis.py                   # Core brain (3800+ lines)
│   ├── JarvisConfig            # Runtime configuration + persistence
│   ├── MemoryStore             # SQLite memory management
│   ├── UserModel               # Adaptive user profiling
│   ├── IntentRouter            # Heuristic-based intent routing
│   ├── IntentClassifierLLM     # LLM-powered intent verification
│   ├── PersonaEngine           # Jarvis tone and persona shaping
│   ├── DialogueTuner           # Fast-path dialogue and quick replies
│   ├── OpenAICompatibleLLMClient  # NVIDIA NIM API client
│   ├── RuleBasedLLMClient      # Local offline fallback
│   ├── LLMOrchestrator         # Multi-model task routing
│   ├── ResearchService         # Web search + Wikipedia + synthesis
│   ├── SystemController        # App launch, URL, browser, clipboard
│   ├── Brain                   # Main orchestration + ingest() pipeline
│   ├── AdvancedBrain           # Voice-capable brain extension
│   ├── HudOverlay              # Tkinter HUD (desktop mode)
│   └── JarvisApp / AdvancedJarvisApp  # App wrappers
│
└── jarvis-webapp/
    ├── server.py               # Python HTTP server (1160 lines)
    │   ├── JarvisRuntime       # Web-facing runtime wrapper
    │   ├── JarvisHandler       # HTTP request routing
    │   │   ├── GET  /          → index.html
    │   │   ├── GET  /api/status → system status JSON
    │   │   ├── GET  /api/location → IP-based location
    │   │   ├── GET  /api/weather → live weather
    │   │   ├── GET  /api/city-news → GDELT city headlines
    │   │   ├── GET  /api/global-news → RSS global headlines
    │   │   ├── GET  /api/market → crypto/stock prices
    │   │   ├── POST /api/command → main AI endpoint
    │   │   ├── POST /api/detection → camera detection data
    │   │   └── POST /api/vision-check → image analysis
    │   └── main()              # ThreadingHTTPServer on port 8080
    │
    ├── index.html              # HUD dashboard structure
    ├── styles.css              # Iron Man-themed dark UI
    ├── app.js                  # Full frontend logic (2500+ lines)
    │   ├── Vision pipeline     # TF.js COCO-SSD, face-api, MobileNet, BLIP
    │   ├── Voice pipeline      # Web Speech API recognition + TTS
    │   ├── Camera management   # getUserMedia, frame capture, periodic scan
    │   ├── Live data            # Weather, news, market auto-fetch
    │   └── Command execution   # API calls, response rendering
    │
    ├── requirements.txt        # requests, beautifulsoup4, psutil, openai
    ├── Procfile                # web: python server.py
    └── .gitignore
```

---

## How to Run

### Prerequisites

- Python 3.10+
- Modern browser (Chrome/Edge recommended for best speech support)

### Quick Start

```powershell
# 1. Navigate to the webapp folder
cd "C:\Users\somya\Documents\jarvis-webapp"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python server.py
```

Open your browser at **http://localhost:8080**

> The server also listens on your local network IP (0.0.0.0), so you can open it on any device on the same WiFi using `http://YOUR_LOCAL_IP:8080`

### Find Your Local IP (for phone/tablet access)

```powershell
ipconfig
# Look for "IPv4 Address" → e.g., 192.168.1.5
# Then open http://192.168.1.5:8080 on any device on your network
```

---

## Configuration

Jarvis loads config from `~/.jarvis/config.json` (auto-created on first run).

| Setting | Default | Description |
|---------|---------|-------------|
| `persona_name` | `Jarvis` | Display name |
| `assistant_tone` | `calm, sharp, witty, detailed` | Persona tone hints |
| `allow_proactive` | `true` | Enables proactive visual announcements |
| `memory_window` | `20` | Number of turns to keep in session context |
| `max_search_results` | `5` | Web search result count |
| `confidence_threshold` | `0.55` | Min confidence to act on intent |
| `learning_enabled` | `true` | Enable user model adaptation |
| `preferred_response_length` | `detailed` | Default response length hint |
| `language` | `en` | Base language |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NVIDIA NIM API key (required for LLM + vision) |
| `PORT` | Server port (default: 8080) |
| `HOST` | Server host (default: 0.0.0.0) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend language | Python 3.10+ (stdlib only + requests + bs4 + openai) |
| HTTP Server | `ThreadingHTTPServer` (stdlib) |
| LLM Provider | NVIDIA NIM API (OpenAI-compatible) |
| LLM Models | DeepSeek V3.1, Llama 3.1 (8B/70B/405B), Nemotron 49B, Llama Vision 11B |
| Vision (Backend) | NVIDIA NIM — llama-3.2-11b-vision-instruct |
| Vision (Frontend) | TensorFlow.js, COCO-SSD, face-api.js, MobileNet, BLIP (WASM) |
| Voice | Web Speech API (browser-native, no server audio) |
| Memory | SQLite (stdlib `sqlite3`) |
| Web Search | DuckDuckGo HTML scrape (no API key) |
| Weather | wttr.in free API |
| News | GDELT v2 API + RSS (BBC, Reuters, Google News, Hacker News) |
| Metals Prices | metals.live + exchangerate-api |
| Frontend | Vanilla HTML + CSS + JavaScript |
| Fonts | Google Fonts (Orbitron, Inter) |

---

<div align="center">

**Built with 🔥 by Somya Bhalani**

*"Sometimes you gotta run before you can walk."*

</div>
