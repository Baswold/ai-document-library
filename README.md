# AI Document Library

A powerful desktop application that lets you **search and chat with ALL your computer files** using local AI models (via Ollama) or cloud API services. Now with **whole-computer cataloging**, glob/grep search, and intelligent file discovery!

## ✨ What's New!

### 🌍 Whole-Computer File Cataloging
- **Index your entire computer** - Automatically scan and catalog ALL files (not just manually added documents)
- **Glob pattern search** - Find files fast with patterns like `*.py`, `*.pdf`, or `test_*.js`
- **Grep-style content search** - Search inside thousands of files instantly with regex support
- **Smart filtering** - Automatically skips system files, temp files, and large binaries
- **Content indexing** - Makes all text files fully searchable (code, logs, configs, etc.)
- **Lightning-fast searches** - Find any file in seconds across your entire computer

### 🔍 Natural Language File Search
Ask questions like:
- "find all Python files"
- "search for files containing 'budget 2024'"
- "show me *.pdf"
- "which files have TODO in them"

## Features

### 🔧 Easy Setup
- **Setup Wizard**: Choose between local AI models or cloud APIs
- **System Detection**: Automatically recommends AI models based on your hardware
- **Ollama Integration**: Built-in support for local AI models
- **API Support**: Ready for OpenAI, Anthropic, and other services

### 🧭 Guided Onboarding
- **Interactive Checklist**: A welcome card tracks the key steps to finish setup
- **Quick Start Guide**: Launch tips and sample questions without leaving the app
- **Dismiss or Revisit**: Hide the helper once you're comfortable, or keep it for shortcuts

### 📄 Document Management
- **Drag & Drop**: Easy document addition
- **Multiple Formats**: PDF, TXT, DOCX, MD support
- **Auto-Cataloging**: AI analyzes each document for content, topics, and type
- **Progress Tracking**: Visual feedback during document processing

### 💬 Intelligent Chat
- **Natural Conversation**: Ask questions about your documents in plain English
- **Context-Aware**: AI finds relevant documents and provides sourced answers
- **Chat History**: Conversation tracking and persistence
- **Real-time Processing**: Background AI processing with visual feedback

### 🔍 Smart Search
- **Content Search**: Find documents by content, not just filename
- **Topic Discovery**: AI-generated topics and themes for each document
- **Relevance Ranking**: Most relevant documents surface first

## Installation

### Prerequisites

1. **Python 3.7+** with tkinter (usually included)
2. **Ollama** (for local AI models) - Download from [ollama.ai](https://ollama.ai)

### Quick Start

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Ollama** (for local AI):
   ```bash
   ollama serve
   ```
4. **Run the application**:
   ```bash
   python main.py
   ```

### First Time Setup

1. **Choose AI Type**: Select "Local Model" (recommended) or "API Service"
2. **For Local Models**:
   - The app will check your system specs
   - Download a recommended model (e.g., `ollama pull gemma2:2b`)
   - Test the connection
3. **For API Services**:
   - Enter your API key (OpenAI/Anthropic)
   - Test the connection
4. **Review the Onboarding Card**:
   - Follow the checklist to add documents and ask your first question
   - Open the Quick Start guide for suggested prompts and usage tips

## Usage

### 🌍 Scanning Your Computer (NEW!)

1. Click **"📂 Scan Computer"** in the left panel
2. Choose what to scan:
   - **Home directory** (Recommended) - Your personal files
   - **Documents folder** - Just documents, downloads, and desktop
   - **Custom directories** - Pick specific folders
3. Click **"Start Scan"** and wait (5-15 minutes typical)
4. See real-time progress: files scanned, indexed, and cataloged
5. Once complete, you can instantly search ALL your files!

### 🔍 Searching Your Files

**Glob Pattern Search** (find by filename):
```
find *.py          - All Python files
show *.pdf         - All PDF files
list *.txt         - All text files
search *.docx      - All Word documents
```

**Content Search** (find text inside files):
```
find files containing "meeting notes"
search for "budget 2024"
which files have "TODO"
grep "import requests"
```

**General Questions**:
```
What Python files do I have?
Find my tax documents
Show me all markdown files
```

### 📚 Adding Documents to Library

1. Click **"+ Add Documents"**
2. Select PDF, TXT, DOCX, or MD files
3. Watch the progress as files are processed and analyzed
4. Documents appear in your library with ✓ when ready

**Note:** Library documents get deep AI analysis and semantic search. Cataloged files are for quick finding!

### 💬 Chatting with Documents

1. Type questions in natural language
2. AI searches your document library
3. Get answers with source references
4. Ask follow-up questions for deeper understanding

### Example Conversations

```
You: "What are the main points in my meeting notes?"
AI: "Based on your meeting notes from Q3-planning.docx, the main points were:
1. Budget allocation for new projects
2. Team restructuring plans
3. Q4 deadlines and milestones..."

You: "Show me documents about machine learning"
AI: "I found 3 documents related to machine learning:
- ML-Basics.pdf: Introduction to machine learning concepts
- Neural-Networks.docx: Deep learning implementation guide
- Data-Science-Notes.txt: Various ML algorithms and examples"
```

## Technical Architecture

### Core Components

- **Main Application** (`main.py`): GUI and orchestration
- **Document Processor** (`document_processor.py`): Text extraction and AI analysis
- **Chat System** (`chat_system.py`): Conversation handling and response generation with glob/grep search
- **File System Cataloger** (`file_system_cataloger.py`): Whole-computer indexing and search
- **Semantic Search** (`semantic_search.py`): Vector embeddings and similarity search
- **SQLite Database**: Document catalog, chat history, and file index

### Document Processing Pipeline

1. **File Ingestion**: Copy documents to managed directory
2. **Text Extraction**: Extract text from various formats
3. **AI Analysis**: Generate summaries, topics, and classification
4. **Chunking**: Split text for optimal search and retrieval
5. **Storage**: Save metadata and chunks to database

### AI Integration

- **Local Models**: Uses Ollama API for privacy and offline operation
- **Cloud APIs**: Ready for OpenAI/Anthropic integration
- **Flexible**: Easy to add new AI providers

## Configuration

Settings are stored in `config.json`:

```json
{
  "ai_type": "local",
  "ollama_url": "http://localhost:11434",
  "local_model": "gemma2:2b",
  "setup_complete": true
}
```

## File Structure

```
ai-document-library/
├── main.py                       # Main application with GUI
├── document_processor.py         # Document text extraction and AI analysis
├── chat_system.py               # Chat functionality with search integration
├── file_system_cataloger.py    # Whole-computer file indexing (NEW!)
├── semantic_search.py           # Vector embeddings and semantic search
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── USER_GUIDE.md                # Detailed user guide (NEW!)
├── CLAUDE.md                    # Project structure documentation
├── TODO.md                      # Development roadmap
├── config.json                  # User configuration (generated)
├── document_library.db          # SQLite database (generated)
└── documents/                   # Managed document storage
```

## Recommended AI Models

### Local Models (via Ollama)

| Model | Size | RAM Required | Best For |
|-------|------|--------------|----------|
| `gemma2:2b` | 1.6GB | 4GB+ | Quick responses, basic analysis |
| `qwen2.5:3b` | 2.2GB | 6GB+ | Better understanding, detailed analysis |
| `llama3.2:3b` | 2.2GB | 6GB+ | Conversational, creative responses |

### Installation
```bash
ollama pull gemma2:2b        # Fastest, most compatible
ollama pull qwen2.5:3b       # Better quality
ollama pull llama3.2:3b      # Most conversational
```

## Troubleshooting

### Common Issues

**"Connection failed" when testing Ollama**
- Ensure Ollama is running: `ollama serve`
- Check the URL: default is `http://localhost:11434`
- Try: `ollama list` to see available models

**"No module named 'PyPDF2'"**
- Install dependencies: `pip install -r requirements.txt`

**Documents not processing**
- Check document format is supported (PDF, TXT, DOCX, MD)
- Ensure Ollama is running and model is downloaded
- Look for error messages in the progress dialog

**Slow responses**
- Local models: Try a smaller model like `gemma2:2b`
- Check available RAM and close other applications
- Consider using cloud APIs for better performance

### Performance Tips

- **For 4GB RAM**: Use `gemma2:2b`
- **For 8GB+ RAM**: Use `qwen2.5:3b` or `llama3.2:3b`
- **Large documents**: Processing may take 30-60 seconds per document
- **Many documents**: Process in smaller batches

## Roadmap

### Recently Completed ✅

- [x] **Whole-Computer File Cataloging**: Index and search entire computer
- [x] **Glob Pattern Search**: Fast file finding with wildcards
- [x] **Grep Content Search**: Search inside thousands of files
- [x] **Smart Filtering**: Automatically skip system/temp files
- [x] **Enhanced Chat**: AI can now search all computer files
- [x] **Enhanced RAG**: Vector embeddings and semantic search

### Planned Features

- [ ] **File System Watcher**: Auto-update catalog when files change
- [ ] **Document Editing**: Edit documents directly in the app
- [ ] **Export Options**: Save conversations and insights
- [ ] **Advanced Filters**: More control over what to index
- [ ] **Cloud Sync**: Optional cloud backup and sync
- [ ] **Plugin System**: Add custom document processors
- [ ] **Web Interface**: Access via browser for remote use

### API Providers

- [ ] **OpenAI**: GPT-4, GPT-3.5 integration
- [ ] **Anthropic**: Claude integration
- [ ] **Local APIs**: Support for other local AI services

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both local and API configurations
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review Ollama documentation for model-specific issues

---

## 🎉 NEW! Whole-Computer Search

This application has been significantly enhanced with **whole-computer file cataloging**! You can now:
- **Index your entire computer** in minutes
- **Search across thousands of files** instantly
- **Use glob patterns** like `*.py` or `*.pdf`
- **Search file contents** with grep-like functionality
- **Chat naturally** - Just ask "find Python files" or "search for budget"

📖 **For detailed usage instructions**, see [USER_GUIDE.md](USER_GUIDE.md)

---

**Happy document chatting!** 🤖📚🔍