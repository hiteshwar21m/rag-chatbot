# MANIT Bhopal RAG Chatbot

An intelligent chatbot for MANIT Bhopal using Retrieval Augmented Generation (RAG) with hybrid retrieval and cross-encoder reranking.

## 🏗️ Architecture Overview

**Technology Stack:**
- **Frontend:** Streamlit
- **Vector Database:** Weaviate (Docker)
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **Reranker:** Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **LLM:** Gemini 2.0 Flash (via OpenRouter API)
- **Web Scraping:** Crawl4AI
- **Document Processing:** Docling

**What is RAG?**

RAG (Retrieval Augmented Generation) combines information retrieval with language models:
1. **Retrieve** relevant documents from a knowledge base
2. **Augment** the LLM prompt with retrieved context
3. **Generate** accurate answers based on actual data

Unlike regular chatbots that rely only on training data, RAG chatbots have access to up-to-date, specific information.

**Our Hybrid Retrieval System:**

- **Path A (Document-level):** Searches document summaries to find relevant documents, then retrieves all chunks from those documents
- **Path B (Chunk-level):** Directly searches individual text chunks
- **Reranking:** Cross-encoder scores all candidates and selects the top 5 most relevant chunks

**Data Flow:**
```
Web Scraping → Cleaning → Chunking → Summarization → 
Deduplication → Embeddings → Weaviate → Chatbot
```

---

## 📁 Project Structure

```
FINAL EXTRACTION_BACKUP/
├── data/                    # All data files
│   ├── raw/                 # Raw scraped content
│   │   ├── webpages/        # Scraped webpage markdown
│   │   └── pdfs/            # Downloaded PDF files
│   ├── extracted/           # Cleaned & processed
│   │   ├── pdf_markdown/    # Converted PDF markdown
│   │   └── webpage_text/    # Cleaned webpage text
│   ├── processed/           # Chunked & embedded data
│   │   ├── chunks_with_metadata.jsonl
│   │   ├── chunks_with_summaries.jsonl
│   │   ├── chunks_final.jsonl
│   │   └── chunks_with_embeddings.jsonl
│   └── urls.txt             # Website URLs (JSON format)
│
├── src/                     # Source code
│   ├── scraping/            # Web scraping scripts
│   │   └── scraper.py       # Crawl4AI webpage scraper
│   ├── download/            # PDF downloaders
│   │   └── pdf_downloader.py
│   ├── cleaning/            # Text cleaning
│   │   └── cleaning_v2.py   # Webpage text cleaner
│   ├── processing/          # Data processing
│   │   └── deduplicate_chunks.py
│   ├── chunking/            # Text chunking
│   │   └── chunking_pipeline.py
│   ├── embedding/           # Vector embeddings
│   │   ├── generate_embeddings.py
│   │   ├── upload_chunks.py
│   │   └── upload_summaries.py
│   ├── summarization/       # AI summaries
│   │   └── summarizer.py
│   ├── retrieval/           # Hybrid search
│   │   └── hybrid_retriever.py
│   ├── interface/           # Streamlit chatbot
│   │   └── chatbot.py
│   ├── monitoring/          # Query logging
│   │   ├── query_logger.py
│   │   └── log_viewer.py
│   ├── evaluation/          # Testing tools
│   │   ├── quick_test.py
│   │   └── evaluate_retrieval.py
│   ├── diagnostics/         # Diagnostic tools
│   │   ├── check_duplicates.py
│   │   └── pdf_diagnostic.py
│   └── config/              # Configuration
│       ├── settings.py      # Config manager
│       └── .env             # Environment variables
│
├── logs/                    # Application logs
├── config/                  # .env file location
├── docker-compose.yml       # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start (For Users with Data)

If you already have the processed data files:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
# Edit config/.env with your OPENROUTER_API_KEY

# 3. Start Weaviate
docker-compose up -d

# 4. Upload data to Weaviate
python src\embedding\upload_chunks.py
python src\embedding\upload_summaries.py

# 5. Run chatbot
streamlit run src\interface\chatbot.py
```

**Access:** Open browser at `http://localhost:8501`

---

## 📦 Complete Setup (From Scratch)

### Prerequisites

- **Docker Desktop** (running)
- **Python 3.8+**
- **OpenRouter API Key** (for LLM)
- Git (optional)

### Step 1: Installation

```powershell
# Clone/download project
cd FINAL_EXTRACTION_BACKUP

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

1. Create a `.env` file in the `config/` folder
2. Add your API key:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```
3. (Optional) Customize other settings - see `config/settings.py` for all available options

### Step 3: Start Weaviate

```powershell
# Start Weaviate with Docker Compose
docker-compose up -d

# Verify it's running
docker ps
# Should show: manit-weaviate container
```

### Step 4: Run Complete Pipeline (OPTIONAL - Data Already Exists!)

> **⚠️ IMPORTANT WARNINGS:**
> 
> **Time-Consuming Processes:**
> - **Web Scraping:** ~30-60 minutes for 189 URLs
> - **PDF Download:** ~2 overnight runs (12-24 hours total)
> - **PDF Extraction (Docling):** ~3-4 overnight runs (36-48 hours total)
> - **Webpage Cleaning:** ~5-10 minutes
> - **Text Chunking:** ~15-20 minutes
> - **Summarization:** ~3-4 hours (25K+ chunks)
> - **Embedding Generation:** ~30-45 minutes
> - **Total:** **~2-3 days** for complete pipeline (mostly overnight runs)
> 
> **API Credit Warnings:**
> - **Summarization uses OpenRouter API** - Requires credits
> - **Free models have severe limitations:**
>   - Only 50 messages per day
>   - Very long queue times (10-30 minutes per request)
>   - Not practical for 25K+ chunks
> - **Recommended:** Use paid tier or skip summarization
> 
> **UPDATE:** All data processing is already complete! The `data/` folder contains:
> - Scraped webpages
> - Cleaned text
> - Generated chunks with summaries
> - Embeddings
> 
> **You can skip to Step 7 (Upload to Weaviate) unless you want to reprocess everything.**

**Complete Pipeline (Only if rebuilding from scratch):**

```powershell
# === DATA COLLECTION (Time: ~2-3 days, mostly overnight) ===

# Step 1: Scrape webpages (~30-60 min for 189 URLs)
python src\scraping\scraper.py

# Step 2: Download PDFs (~2 overnight runs: 12-24 hours total)
# Note: pdf_downloader.py not fully integrated yet
# python src\download\pdf_downloader.py

# Step 3: Extract PDFs to Markdown using Docling (~3-4 overnight runs: 36-48 hours)
# This converts PDFs to markdown format
# Manual process - PDFs already extracted in data/extracted/pdf_markdown/

# === TEXT PROCESSING (Time: ~20-30 min) ===

# Step 4: Clean webpages (~5-10 min)
python src\cleaning\cleaning_v2.py

# Step 5: Chunk all documents (PDFs + webpages) (~15-20 min)
python src\chunking\chunking_pipeline.py

# === AI PROCESSING (Time: ~4-5 hours, Requires API Credits!) ===

# Step 6: Generate summaries (~3-4 hours, USES API CREDITS!)
# ⚠️ Warning: This will use OpenRouter API credits
# ⚠️ Free models: 50 msg/day limit + long queues = impractical
python src\summarization\summarizer.py

# Step 7: Deduplicate chunks (~1-2 min)
python src\processing\deduplicate_chunks.py

# === EMBEDDING & UPLOAD (Time: ~30-45 min) ===

# Step 8: Generate embeddings (~30-45 min)
python src\embedding\generate_embeddings.py

# Step 9: Upload to Weaviate (~2-3 min)
python src\embedding\upload_chunks.py
python src\embedding\upload_summaries.py

# === RUN CHATBOT ===

# Step 10: Launch chatbot
streamlit run src\interface\chatbot.py
```

**Recommended Path for New Users:**

Skip Steps 1-8 (data already exists) and just run:

```powershell
# Upload existing data
python src\embedding\upload_chunks.py
python src\embedding\upload_summaries.py

# Run chatbot
streamlit run src\interface\chatbot.py
```

---

## ⚙️ Configuration

All settings are in `config/.env`:

**Required:**
- `OPENROUTER_API_KEY` - Your OpenRouter API key

**LLM Settings:**
- `LLM_MODEL` - Model name (default: google/gemini-2.0-flash-lite-001)
- `LLM_TEMPERATURE` - Response randomness (default: 0.3)

**Embedding Models:**
- `EMBEDDING_MODEL` - For chunk embeddings
- `RERANKER_MODEL` - For reranking

**Weaviate:**
- `WEAVIATE_HOST` - Default: localhost
- `WEAVIATE_HTTP_PORT` - Default: 8080
- `WEAVIATE_GRPC_PORT` - Default: 50051

**File Paths:**
- See `.env` for all configurable paths

---

## 🧪 Testing & Diagnostics

**Test retrieval:**
```powershell
python src\evaluation\quick_test.py
```

**Check for duplicates:**
```powershell
python src\diagnostics\check_duplicates.py
```

**View query logs:**
```powershell
python src\monitoring\log_viewer.py
```

---

## 🔧 Docker Management

**Start Weaviate:**
```powershell
docker-compose up -d
```

**Stop Weaviate:**
```powershell
docker-compose down
```

**View logs:**
```powershell
docker-compose logs -f weaviate
```

**Restart (preserves data):**
```powershell
docker-compose restart
```

---

## 📊 Data Statistics

- **Webpage URLs:** 189 across 23 categories
- **Total Chunks:** ~25,055
- **Unique Documents:** ~1,432
- **Embedding Dimension:** 384

---

## 🐛 Troubleshooting

**Docker connection failed:**
- Ensure Docker Desktop is running
- Check ports 8080 and 50051 are not in use

**Port conflicts:**
```powershell
# Stop existing containers
docker stop manit-weaviate
```

**API key errors:**
- Verify `OPENROUTER_API_KEY` in `config/.env`
- Check API key is valid and has credits

**Missing dependencies:**
```powershell
# Reinstall
pip install -r requirements.txt --upgrade
```

**Weaviate upload errors:**
- Check GRPC message size in docker-compose.yml (should be 104857600)
- Restart Weaviate: `docker-compose restart`

---
