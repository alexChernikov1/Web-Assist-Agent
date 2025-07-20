# Web Assist Agent (Firecrawl + DeepSeek Enhanced)

## Description

**Web Assist Agent** is an AI-powered research and Q&A assistant focused on appliance parts compatibility, powered by Firecrawl web scraping and DeepSeek LLMs. It features a FastAPI backend that intelligently identifies appliance part and model numbers, scrapes relevant product pages, caches results, and answers user queries through a local API endpoint. Optional React frontend allows chat-based interaction.

## Features

- 🔥 **Firecrawl-Powered Research**: Scrapes live product pages on partselect.com with rate-limited retry handling.
- 🤖 **DeepSeek LLM API Integration**: Uses DeepSeek API to generate responses with in-context information.
- 🧠 **Compatibility Classification**: Detects compatibility questions via regex and fallback LLM classification.
- 📑 **Persistent HTML Caching**: Caches visited links, model/part HTML, and compatibility data to avoid redundant scraping.
- 📊 **Flow Graph Building**: (Optional) Visualizes data flow and caching.
- 🖥️ **FastAPI Backend**: Provides `/chat` endpoint for querying the agent via API.
- 🌐 **React Frontend (Optional)**: Interactive chat UI served at `localhost:3000`.

## Installation

### Backend Setup

```bash
git clone https://github.com/alexChernikov1/Web-Assist-Agent.git
cd Web-Assist-Agent
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ILbackend/requirements.txt
```

### API Keys

Create a `python-dotenv.env` file in the project root:

```
DEEP_SEEK_WEB_API_KEY=your-deepseek-key
FIRECRAWL_API_KEY=your-firecrawl-key
```

### Frontend Setup (Optional)

```bash
cd ILfrontend
npm install
```

## Usage

### Backend (FastAPI)

```bash
cd ILbackend
python main.py
```

Backend will run on `http://localhost:8000/chat`.

### Frontend

```bash
cd ILfrontend
npm start
```

Frontend runs on `http://localhost:3000`.

## How It Works

- Extracts part/model numbers from questions.
- Uses cached links or scrapes partselect.com via Firecrawl.
- Detects compatibility context and scrapes model pages for compatible parts.
- Answers via DeepSeek LLM using custom prompt templates.
- Stores browsing history and caches for efficiency.

## Technologies Used

- Python (FastAPI, Selenium, OpenAI, Firecrawl, Pydantic, dotenv)
- DeepSeek LLM API
- Firecrawl API (scraping + search)
- React (Frontend)
- NetworkX/Matplotlib (Optional visualization)
- Docker-ready architecture (self-hostable)

## License

Licensed under **MIT License**. See LICENSE.
