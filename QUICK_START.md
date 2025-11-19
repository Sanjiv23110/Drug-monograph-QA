# 🚀 Quick Start Guide - Medical RAG System

## ⚡ 3-Step Quick Start (5 minutes)

### Step 1: Verify Everything Works

```bash
python test_system.py
```

**Expected output**: "✓ All tests passed! System is ready to use."

### Step 2: Try the Demo

```bash
python demo.py
```

**What this does**: Processes your sample PDF and asks demo questions

### Step 3: Choose Your Interface

#### Option A: Web Interface (Easiest)

```bash
python web_interface.py
```

Then open: **http://127.0.0.1:7860** in your browser

#### Option B: Command Line

```bash
python main.py --interactive
```

---

## 🎯 Ready to Use Commands

### Upload and Process PDFs

```bash
# Single file
python main.py --pdf-file "data/your_document.pdf"

# All PDFs in folder
python main.py --pdf-dir "data/"
```

### Ask Questions

```bash
# Single question
python main.py --question "What are the side effects of this medication?"

# Interactive chat
python main.py --interactive
```

### Check System Status

```bash
python main.py --stats
```

---

## 📁 File Organization

- Put your PDF files in: `data/` folder
- System creates: `logs/`, `vector_db/`, `models/` automatically

---

## 🔧 If Something Goes Wrong

### Elasticsearch not running:

```bash
docker-compose up -d elasticsearch
```

### Import errors:

```bash
pip install -r requirements.txt
```

### Clear everything and start fresh:

```bash
python main.py --clear
```

---

## 🎉 You're Ready!

Your Medical RAG System is working! The demo just processed 107 text chunks from your PDF and answered questions successfully.

**Next**: Try the web interface for the easiest experience!
