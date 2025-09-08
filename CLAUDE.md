# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This Documents directory contains various personal projects and learning materials, with several active coding projects:

### Main Projects

- **GitHub/ai-dreaming2/**: Autonomous AI reasoning system (Python, Docker)
- **python_dictanary/**: Python function dictionary app (tkinter GUI)
- **Science and computers/**: Arduino projects, Python experiments, and learning code

## Key Development Commands

### Python Dictionary App
```bash
cd python_dictanary
python3 python_dictionary_app.py
```

### AI Dreaming Project
```bash
cd GitHub/ai-dreaming2
# Install dependencies
pip install requests
# Start Ollama service (required)
ollama serve
# Run the application
python3 dreaming_ai.py
```

### Arduino Projects
Located in `Science and computers/Arduino/` - use Arduino IDE for development.

## Project Structure

- **GitHub/**: Git repositories and collaborative projects
- **python_dictanary/**: Standalone Python dictionary application
- **Science and computers/**: Educational and experimental code
  - Arduino sketches and libraries
  - Python learning projects
  - Machine learning experiments
- **ai_libry/**: AI-related resources
- **screen shots/**: Documentation and reference images

## File Formats and Dependencies

### Python Projects
- Standard Python 3.7+ with tkinter
- AI projects require Ollama for local LLM inference
- No package.json or formal dependency management outside requirements.txt files

### Arduino Projects
- Standard Arduino IDE environment
- Hardware-specific libraries in `Science and computers/Arduino/libraries/`

## No Centralized Testing

Individual projects may have their own testing approaches:
- ai-dreaming2: No formal tests (experimental AI system)
- python_dictanary: Manual testing via GUI
- Arduino projects: Hardware-dependent testing

## Important Notes

- This is a personal learning and experimentation directory
- Projects range from educational experiments to functional applications
- Each subdirectory may have its own README or documentation
- Many projects are works-in-progress or learning exercises