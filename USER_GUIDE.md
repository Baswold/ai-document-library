# AI Document Library - User Guide

## 🌟 Welcome!

AI Document Library is your personal AI assistant that helps you **search, organize, and chat with ALL your computer files**. No more forgetting where you saved that important document or what was in that file from last year!

---

## ✨ What's New - Computer-Wide Search!

The app can now:
- **📂 Catalog your entire computer** - Not just manually added documents, but ALL your files!
- **🔍 Find files instantly** - Use powerful glob patterns like `*.py` or `*.pdf`
- **🔎 Search inside files** - Find any text across thousands of files in seconds
- **💬 Chat naturally** - Ask "find files containing 'budget'" or "show me all Python files"
- **🤖 Smart filtering** - Automatically skips system files, temporary files, and large binaries

---

## 🚀 Getting Started

### First Time Setup

1. **Choose Your AI Model**
   - **Local Model (Recommended)**: Free, private, works offline
     - Requires Ollama to be installed and running
     - Download from: https://ollama.com
   - **Cloud API**: Fast and powerful, but costs money and sends data online

2. **Scan Your Computer**
   - Click **"📂 Scan Computer"** in the left panel
   - Choose what to scan:
     - **Home directory** (Recommended) - Scans your personal files
     - **Documents folder** - Just your documents, downloads, and desktop
     - **Custom directories** - Pick specific folders
   - Wait for the scan to complete (may take 5-15 minutes depending on file count)

3. **Start Chatting!**
   - Ask questions about your files
   - Search for documents
   - Get summaries and insights

---

## 💬 How to Use - Example Questions

### Finding Files

**Glob Pattern Searches** (find files by name):
```
find *.pdf
show me all *.py files
list *.txt
search for *.docx
```

**Content Searches** (find text inside files):
```
find files containing "meeting notes"
search for "budget 2024"
which files have "TODO"
grep "import requests"
```

**General Questions**:
```
What documents do I have about machine learning?
Find my tax documents
Show me Python files about web scraping
What's in my meeting notes from last week?
```

---

## 📚 Two Ways to Add Documents

### 1. Scan Your Computer (NEW!)
- Automatically indexes ALL your files
- Updates when you re-scan
- Fast searches across your entire computer
- **Best for:** Finding any file on your computer

### 2. Add Documents to Library
- Click "**+ Add Documents**"
- Manually select important documents
- Gets deep AI analysis and summaries
- Stored in a local library with embeddings
- **Best for:** Documents you want AI insights about

💡 **Pro Tip:** Use both! Scan your computer for finding files, and add important documents to the library for detailed AI analysis.

---

## 🎯 Key Features

### 1. Computer-Wide File Catalog
- **Indexes millions of files** in minutes
- **Smart filtering** - Skips system files, binaries, and temp files
- **Content indexing** - Makes text files fully searchable (code, markdown, logs, etc.)
- **Fast searches** - Find anything in seconds

### 2. Glob Pattern Search
Search by file patterns:
- `*.py` - All Python files
- `*.md` - All Markdown files
- `*.pdf` - All PDFs
- `test_*.py` - Python test files
- `*report*.docx` - Word docs with "report" in the name

### 3. Grep Content Search
Search inside files:
- Plain text search: `"import numpy"`
- Case-sensitive or insensitive
- Shows matching lines with line numbers
- Searches code, logs, documents, and more

### 4. AI Chat Assistant
- Ask questions in plain English
- Get intelligent responses based on your files
- References specific documents in answers
- Understands context from previous messages

### 5. Document Library
- Manually add important documents
- AI generates summaries and extracts topics
- Semantic search with embeddings
- Ask questions specific to your document collection

---

## 🔧 Configuration

### Scan Settings

Click **"⚙️ Configure"** to customize:
- Which directories to scan
- Choose from presets or custom paths
- Re-run scans to update the catalog

### Skip Directories

The scanner automatically skips:
- System directories (`/proc`, `/sys`, `C:\Windows`)
- Hidden directories (`.git`, `.cache`, `node_modules`)
- Virtual environments (`venv`, `env`)
- Large binary directories

### File Types Indexed

**Fully searchable (content indexed):**
- Code files (`.py`, `.js`, `.java`, `.cpp`, `.go`, etc.)
- Documents (`.txt`, `.md`, `.rst`, `.log`)
- Config files (`.json`, `.yaml`, `.toml`, `.ini`)
- Data files (`.csv`, `.sql`)
- Web files (`.html`, `.css`)

**Cataloged (filename/path searchable):**
- PDFs, Word docs, images, videos, etc.
- All other files

---

## 🛠️ Advanced Usage

### Regex Content Search

Enable regex mode for powerful pattern matching:
```python
# In Python code, modify grep_search call:
results = file_catalog.grep_search(
    pattern=r"def\s+\w+\s*\(",  # Find function definitions
    regex=True,
    case_sensitive=False
)
```

### Custom Scan Filters

Add custom directories to skip or file extensions to ignore by modifying the `FileSystemCatalog` configuration in your code.

### API Integration

The chat system supports multiple AI providers:
- **Ollama** (Local) - Free and private
- **OpenAI** (Cloud) - GPT-4, GPT-3.5
- **Anthropic** (Cloud) - Claude

---

## 📊 Statistics

View your catalog stats in the left panel:
- **Total files indexed**
- **Total size** of cataloged files
- **Content-indexed files** (fully searchable)
- **Number of directories**

---

## 🔒 Privacy & Security

- **Local-first** - All data stays on your computer
- **No cloud uploads** - Your files never leave your machine (when using local AI)
- **SQLite database** - All indexes stored locally
- **Smart filtering** - Automatically avoids sensitive system files

---

## ❓ Troubleshooting

### Scan is Slow
- Normal! First scan can take 10-20 minutes for 100,000+ files
- Subsequent scans are faster (only checks changed files)
- Consider scanning fewer directories

### Not Finding Files
- Make sure you've completed a scan first
- Try different search terms
- Check if the directory containing the file was included in scan

### AI Not Responding
- **Local model:** Make sure Ollama is running (`ollama serve`)
- **Check model name** in config matches installed model
- **API users:** Verify API key is correct

### Out of Memory
- Reduce scan directories
- The scanner automatically skips very large files (>100MB)
- Consider more RAM for very large catalogs

---

## 🎓 Tips & Tricks

1. **Re-scan regularly** - Run a scan monthly to keep catalog fresh
2. **Use specific search terms** - "budget_2024.xlsx" vs "budget"
3. **Combine with library** - Add frequently-used docs to library for better AI insights
4. **Try different questions** - The AI understands many phrasings
5. **Use glob for quick finds** - `*.pdf` is faster than asking AI

---

## 🚀 Performance Tips

- **First scan:** Be patient! Indexing takes time
- **Regular scans:** Much faster, only updates changed files
- **Semantic search:** Requires additional memory for embeddings
- **Large catalogs:** 1M+ files may need 16GB+ RAM

---

## 📝 Technical Details

### File System Cataloger
- **Language:** Python 3.7+
- **Database:** SQLite (fast, embedded)
- **Indexing:** Multi-threaded for performance
- **Search:** Glob (fnmatch) and Regex (re) based
- **Smart filtering:** Configurable skip lists

### Search Capabilities
- **Glob search:** Fast filename matching
- **Grep search:** Content regex with context
- **Combined search:** Filename + path + content
- **Semantic search:** ML-based similarity (document library)

### AI Integration
- **Local:** Ollama with any model (llama, gemma, qwen, etc.)
- **Cloud:** OpenAI, Anthropic (API keys required)
- **RAG:** Retrieval-Augmented Generation for accurate answers

---

## 🤝 Need Help?

- Check the README.md for installation instructions
- Review TODO.md for planned features
- See CLAUDE.md for project structure details

---

## 🎉 Enjoy Your Enhanced AI Document Library!

Now you can find ANY file on your computer and chat about all your documents! No more "where did I save that file?" moments. Just ask your AI assistant! 🚀
