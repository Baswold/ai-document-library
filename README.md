# AI Document Library

A desktop application that lets you chat with your own document collection using either local AI models (via Ollama) or cloud API services.

## Features

### 🔧 Easy Setup
- **Setup Wizard**: Choose between local AI models or cloud APIs
- **System Detection**: Automatically recommends AI models based on your hardware
- **Ollama Integration**: Built-in support for local AI models
- **API Support**: Ready for OpenAI, Anthropic, and other services

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

## Usage

### Adding Documents

1. Click **"+ Add Documents"**
2. Select PDF, TXT, DOCX, or MD files
3. Watch the progress as files are processed and analyzed
4. Documents appear in your library with ✓ when ready

### Chatting with Documents

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
- **Chat System** (`chat_system.py`): Conversation handling and response generation
- **SQLite Database**: Document catalog and chat history

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
├── main.py                    # Main application
├── document_processor.py      # Document analysis
├── chat_system.py            # Chat functionality
├── requirements.txt          # Dependencies
├── README.md                # This file
├── config.json              # User configuration
├── document_library.db      # SQLite database
└── documents/               # Managed document storage
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

### Planned Features

- [ ] **Enhanced RAG**: Vector embeddings and semantic search
- [ ] **Document Editing**: Edit documents directly in the app
- [ ] **Export Options**: Save conversations and insights
- [ ] **Batch Processing**: Process entire folders at once
- [ ] **Cloud Sync**: Optional cloud backup and sync
- [ ] **Plugin System**: Add custom document processors

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

**Happy document chatting!** 🤖📚