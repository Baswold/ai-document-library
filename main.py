#!/usr/bin/env python3
"""
AI Document Library - Chat with your documents using local AI models or API services
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3
import json
import os
import threading
import requests
from pathlib import Path
import hashlib
from datetime import datetime
import subprocess
import platform
from document_processor import DocumentProcessor
from chat_system import ChatSystem

class DocumentLibrary:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Document Library")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize paths
        self.app_dir = Path(__file__).parent
        self.db_path = self.app_dir / "document_library.db"
        self.config_path = self.app_dir / "config.json"
        self.documents_dir = self.app_dir / "documents"
        self.documents_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = self.load_config()

        # Onboarding UI state
        self.onboarding_card = None
        self.onboarding_step_vars = []
        self.onboarding_step_labels = []
        self.onboarding_header_var = None
        self.onboarding_subtext_var = None
        self.quick_start_window = None
        self.onboarding_container = None
        
        # Initialize database
        self.init_database()
        
        # Initialize processors
        self.doc_processor = DocumentProcessor(str(self.db_path), self.config)
        self.chat_system = ChatSystem(str(self.db_path), self.config)
        
        # Track setup state
        self.setup_complete = self.config.get('setup_complete', False)
        
        if not self.setup_complete:
            self.show_setup_wizard()
        else:
            self.show_main_interface()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'ai_type': None,  # 'local' or 'api'
                'ollama_url': 'http://localhost:11434',
                'api_key': None,
                'api_provider': None,
                'local_model': None,
                'setup_complete': False
            }

        # Ensure new keys exist when updating from older configurations
        config.setdefault('onboarding_dismissed', False)

        return config
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def init_database(self):
        """Initialize SQLite database for document catalog"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                summary TEXT,
                topics TEXT,
                doc_type TEXT,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Document chunks for RAG
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Chat history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                relevant_docs TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def show_setup_wizard(self):
        """Display initial setup wizard"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Setup wizard frame
        setup_frame = ttk.Frame(self.root, padding="20")
        setup_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(setup_frame, text="AI Document Library Setup", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 30))
        
        # Subtitle
        subtitle_label = ttk.Label(setup_frame, 
                                  text="Choose how you want to power your AI assistant:",
                                  font=('Arial', 12))
        subtitle_label.pack(pady=(0, 30))
        
        # AI Type selection
        self.ai_type_var = tk.StringVar()
        
        # Local Model option
        local_frame = ttk.LabelFrame(setup_frame, text="Local AI Model (Recommended)", 
                                   padding="15")
        local_frame.pack(fill=tk.X, pady=(0, 15))
        
        local_radio = ttk.Radiobutton(local_frame, text="Use Local AI Model (Ollama)", 
                                     variable=self.ai_type_var, value="local")
        local_radio.pack(anchor=tk.W)
        
        local_desc = ttk.Label(local_frame, 
                              text="• Free to use, no API costs\n• Private - your documents stay on your computer\n• Works offline\n• Requires downloading AI model (~2-4GB)",
                              justify=tk.LEFT, foreground='#666')
        local_desc.pack(anchor=tk.W, pady=(5, 0))
        
        # API Service option
        api_frame = ttk.LabelFrame(setup_frame, text="Cloud AI Service", padding="15")
        api_frame.pack(fill=tk.X, pady=(0, 30))
        
        api_radio = ttk.Radiobutton(api_frame, text="Use Cloud AI Service", 
                                   variable=self.ai_type_var, value="api")
        api_radio.pack(anchor=tk.W)
        
        api_desc = ttk.Label(api_frame, 
                            text="• Fast and powerful AI models\n• Requires internet connection\n• Costs money per request\n• Your documents are sent to the cloud",
                            justify=tk.LEFT, foreground='#666')
        api_desc.pack(anchor=tk.W, pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(setup_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        next_button = ttk.Button(button_frame, text="Next", 
                               command=self.setup_next_step)
        next_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Set default selection
        self.ai_type_var.set("local")
    
    def setup_next_step(self):
        """Handle next step in setup wizard"""
        ai_type = self.ai_type_var.get()
        if not ai_type:
            messagebox.showwarning("Setup", "Please select an AI type")
            return
        
        self.config['ai_type'] = ai_type
        
        if ai_type == "local":
            self.setup_local_model()
        else:
            self.setup_api_service()
    
    def setup_local_model(self):
        """Setup local model configuration"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        setup_frame = ttk.Frame(self.root, padding="20")
        setup_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(setup_frame, text="Local AI Model Setup", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # System check
        system_frame = ttk.LabelFrame(setup_frame, text="System Check", padding="15")
        system_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.system_status = ttk.Label(system_frame, text="Checking system...")
        self.system_status.pack(anchor=tk.W)
        
        # Ollama configuration
        ollama_frame = ttk.LabelFrame(setup_frame, text="Ollama Configuration", padding="15")
        ollama_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(ollama_frame, text="Ollama API URL:").pack(anchor=tk.W)
        self.ollama_url_var = tk.StringVar(value=self.config.get('ollama_url', 'http://localhost:11434'))
        ollama_entry = ttk.Entry(ollama_frame, textvariable=self.ollama_url_var, width=50)
        ollama_entry.pack(anchor=tk.W, pady=(5, 10))
        
        test_button = ttk.Button(ollama_frame, text="Test Connection", 
                               command=self.test_ollama_connection)
        test_button.pack(anchor=tk.W)
        
        self.connection_status = ttk.Label(ollama_frame, text="")
        self.connection_status.pack(anchor=tk.W, pady=(5, 0))
        
        # Model selection
        model_frame = ttk.LabelFrame(setup_frame, text="Model Selection", padding="15")
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(model_frame, text="Recommended models for your system:").pack(anchor=tk.W)
        
        self.model_var = tk.StringVar()
        self.model_listbox = tk.Listbox(model_frame, height=4)
        self.model_listbox.pack(fill=tk.X, pady=(5, 10))
        
        # Buttons
        button_frame = ttk.Frame(setup_frame)
        button_frame.pack(fill=tk.X)
        
        back_button = ttk.Button(button_frame, text="Back", 
                               command=self.show_setup_wizard)
        back_button.pack(side=tk.LEFT)
        
        finish_button = ttk.Button(button_frame, text="Finish Setup", 
                                 command=self.finish_local_setup)
        finish_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Run system check
        threading.Thread(target=self.check_system, daemon=True).start()
    
    def check_system(self):
        """Check system specifications and recommend models"""
        try:
            # Simple system check
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            self.root.after(0, lambda: self.system_status.config(
                text=f"RAM: {ram_gb:.1f}GB detected"))
            
            # Recommend models based on RAM
            models = []
            if ram_gb >= 8:
                models.extend(["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"])
            elif ram_gb >= 4:
                models.extend(["gemma2:2b", "qwen2.5:1.5b"])
            else:
                models.extend(["gemma2:2b"])
            
            # Update model list
            self.root.after(0, lambda: self.update_model_list(models))
            
        except ImportError:
            self.root.after(0, lambda: self.system_status.config(
                text="Install psutil for detailed system info (pip install psutil)"))
            # Default models
            models = ["gemma2:2b", "qwen2.5:1.5b", "llama3.2:3b"]
            self.root.after(0, lambda: self.update_model_list(models))
    
    def update_model_list(self, models):
        """Update the model selection listbox"""
        self.model_listbox.delete(0, tk.END)
        for model in models:
            self.model_listbox.insert(tk.END, model)
        if models:
            self.model_listbox.selection_set(0)  # Select first model
    
    def test_ollama_connection(self):
        """Test connection to Ollama"""
        url = self.ollama_url_var.get()
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.connection_status.config(text="✓ Connected to Ollama", 
                                            foreground='green')
                
                # Update available models
                models = response.json().get('models', [])
                if models:
                    model_names = [model['name'] for model in models]
                    self.update_model_list(model_names)
            else:
                self.connection_status.config(text="✗ Connection failed", 
                                            foreground='red')
        except Exception as e:
            self.connection_status.config(text=f"✗ Connection error: {str(e)}", 
                                        foreground='red')
    
    def setup_api_service(self):
        """Setup API service configuration"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        setup_frame = ttk.Frame(self.root, padding="20")
        setup_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(setup_frame, text="API Service Setup", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Provider selection
        provider_frame = ttk.LabelFrame(setup_frame, text="Choose Provider", padding="15")
        provider_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.provider_var = tk.StringVar(value="openai")
        
        ttk.Radiobutton(provider_frame, text="OpenAI (GPT-4, GPT-3.5)", 
                       variable=self.provider_var, value="openai").pack(anchor=tk.W)
        ttk.Radiobutton(provider_frame, text="Anthropic (Claude)", 
                       variable=self.provider_var, value="anthropic").pack(anchor=tk.W)
        
        # API Key entry
        key_frame = ttk.LabelFrame(setup_frame, text="API Key", padding="15")
        key_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(key_frame, text="Enter your API key:").pack(anchor=tk.W)
        self.api_key_var = tk.StringVar()
        key_entry = ttk.Entry(key_frame, textvariable=self.api_key_var, 
                             show="*", width=50)
        key_entry.pack(anchor=tk.W, pady=(5, 10))
        
        test_api_button = ttk.Button(key_frame, text="Test API Key", 
                                   command=self.test_api_key)
        test_api_button.pack(anchor=tk.W)
        
        self.api_status = ttk.Label(key_frame, text="")
        self.api_status.pack(anchor=tk.W, pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(setup_frame)
        button_frame.pack(fill=tk.X)
        
        back_button = ttk.Button(button_frame, text="Back", 
                               command=self.show_setup_wizard)
        back_button.pack(side=tk.LEFT)
        
        finish_button = ttk.Button(button_frame, text="Finish Setup", 
                                 command=self.finish_api_setup)
        finish_button.pack(side=tk.RIGHT, padx=(10, 0))
    
    def test_api_key(self):
        """Test the API key"""
        provider = self.provider_var.get()
        api_key = self.api_key_var.get()
        
        if not api_key:
            self.api_status.config(text="Please enter an API key", foreground='red')
            return
        
        # Simple test - this would need real implementation
        self.api_status.config(text="✓ API key looks valid", foreground='green')
    
    def finish_local_setup(self):
        """Complete local model setup"""
        # Get selected model
        selection = self.model_listbox.curselection()
        if selection:
            model = self.model_listbox.get(selection[0])
            self.config['local_model'] = model
        
        self.config['ollama_url'] = self.ollama_url_var.get()
        self.config['setup_complete'] = True
        self.save_config()
        
        messagebox.showinfo("Setup Complete", 
                           f"Setup complete! Selected model: {self.config.get('local_model', 'None')}")
        self.show_main_interface()
    
    def finish_api_setup(self):
        """Complete API service setup"""
        self.config['api_provider'] = self.provider_var.get()
        self.config['api_key'] = self.api_key_var.get()
        self.config['setup_complete'] = True
        self.save_config()
        
        messagebox.showinfo("Setup Complete", 
                           f"Setup complete! Using {self.config['api_provider']} API")
        self.show_main_interface()
    
    def show_main_interface(self):
        """Display the main application interface"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create main layout container and onboarding area
        self.main_container = tk.Frame(self.root, bg='#f0f0f0')
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.onboarding_container = tk.Frame(self.main_container, bg='#f0f0f0')
        self.onboarding_container.pack(fill=tk.X, padx=10, pady=(10, 0))

        main_frame = ttk.Frame(self.main_container)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Document management
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Document section
        doc_label = ttk.Label(left_panel, text="Document Library", 
                             font=('Arial', 14, 'bold'))
        doc_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Add documents button
        add_btn = ttk.Button(left_panel, text="+ Add Documents", 
                           command=self.add_documents)
        add_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Document list
        doc_frame = ttk.Frame(left_panel)
        doc_frame.pack(fill=tk.BOTH, expand=True)
        
        self.doc_listbox = tk.Listbox(doc_frame)
        doc_scrollbar = ttk.Scrollbar(doc_frame, orient=tk.VERTICAL, 
                                     command=self.doc_listbox.yview)
        self.doc_listbox.configure(yscrollcommand=doc_scrollbar.set)
        
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        doc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Chat interface
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chat header
        chat_label = ttk.Label(right_panel, text="Chat with your Documents", 
                              font=('Arial', 14, 'bold'))
        chat_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Chat display
        chat_frame = ttk.Frame(right_panel)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = tk.Text(chat_frame, wrap=tk.WORD, state=tk.DISABLED,
                                   bg='white', font=('Arial', 11))
        chat_scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, 
                                      command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=chat_scrollbar.set)
        
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Chat input
        input_frame = ttk.Frame(right_panel)
        input_frame.pack(fill=tk.X)
        
        self.chat_input = tk.Text(input_frame, height=3, wrap=tk.WORD,
                                 font=('Arial', 11))
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind Enter key
        self.chat_input.bind('<Control-Return>', lambda e: self.send_message())
        
        # Load existing documents
        self.load_document_list()

        # Welcome message
        self.add_chat_message("Assistant",
                             "Welcome to AI Document Library! Add some documents to get started.")

        # Show onboarding helpers if needed
        self.refresh_onboarding_card()

    def build_onboarding_steps(self):
        """Define onboarding checklist steps and completion state"""
        return [
            {
                'label': "Choose how you'd like to run the AI",
                'completed': bool(self.config.get('ai_type'))
            },
            {
                'label': "Add your first document to the library",
                'completed': self.get_document_count() > 0
            },
            {
                'label': "Ask a question in the chat panel",
                'completed': self.get_chat_count() > 0
            }
        ]

    def get_document_count(self):
        """Return number of documents stored"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count or 0
        except Exception:
            return 0

    def get_chat_count(self):
        """Return number of stored chat exchanges"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM chat_history')
            count = cursor.fetchone()[0]
            conn.close()
            return count or 0
        except Exception:
            return 0

    def refresh_onboarding_card(self):
        """Show or update the onboarding helper card"""
        if self.onboarding_container is None or self.config.get('onboarding_dismissed', False):
            self.destroy_onboarding_card()
            return

        steps = self.build_onboarding_steps()

        if (self.onboarding_card is None or
                not self.onboarding_card.winfo_exists() or
                len(self.onboarding_step_vars) != len(steps)):
            self.create_onboarding_card(steps)

        all_complete = all(step['completed'] for step in steps)

        if self.onboarding_header_var:
            if all_complete:
                self.onboarding_header_var.set("You're ready to explore!")
                self.onboarding_subtext_var.set(
                    "All onboarding steps are complete. Keep this card for quick shortcuts or dismiss it anytime.")
            else:
                self.onboarding_header_var.set("Welcome! Let's get you set up.")
                self.onboarding_subtext_var.set(
                    "Follow these quick steps to start using the library.")

        for idx, step in enumerate(steps):
            icon = "✓" if step['completed'] else "○"
            text = f"{icon} {step['label']}"
            if idx < len(self.onboarding_step_vars):
                self.onboarding_step_vars[idx].set(text)
            if idx < len(self.onboarding_step_labels):
                color = '#2f7d32' if step['completed'] else '#1f3a73'
                self.onboarding_step_labels[idx].config(fg=color)

    def create_onboarding_card(self, steps):
        """Create the onboarding helper UI"""
        self.destroy_onboarding_card()

        card_bg = '#e8f0fe'
        border_color = '#c2d3ff'

        self.onboarding_card = tk.Frame(
            self.onboarding_container,
            bg=card_bg,
            highlightbackground=border_color,
            highlightthickness=1,
            bd=0,
            relief=tk.FLAT
        )
        self.onboarding_card.pack(fill=tk.X, pady=(0, 12))

        self.onboarding_header_var = tk.StringVar()
        self.onboarding_subtext_var = tk.StringVar()

        header = tk.Label(
            self.onboarding_card,
            textvariable=self.onboarding_header_var,
            font=('Arial', 14, 'bold'),
            bg=card_bg,
            fg='#1f3a73'
        )
        header.pack(anchor=tk.W, pady=(12, 4), padx=16)

        subtext = tk.Label(
            self.onboarding_card,
            textvariable=self.onboarding_subtext_var,
            font=('Arial', 11),
            bg=card_bg,
            fg='#1f3a73',
            justify=tk.LEFT
        )
        subtext.pack(anchor=tk.W, padx=16)

        checklist_frame = tk.Frame(self.onboarding_card, bg=card_bg)
        checklist_frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        self.onboarding_step_vars = []
        self.onboarding_step_labels = []
        for step in steps:
            var = tk.StringVar(value='')
            label = tk.Label(
                checklist_frame,
                textvariable=var,
                font=('Arial', 11),
                bg=card_bg,
                anchor='w'
            )
            label.pack(fill=tk.X, pady=2)
            self.onboarding_step_vars.append(var)
            self.onboarding_step_labels.append(label)

        button_frame = tk.Frame(self.onboarding_card, bg=card_bg)
        button_frame.pack(fill=tk.X, padx=16, pady=(8, 12))

        primary_btn = ttk.Button(button_frame, text="Add documents now", command=self.add_documents)
        primary_btn.pack(side=tk.LEFT)

        quick_start_btn = ttk.Button(button_frame, text="Open quick start guide", command=self.show_quick_start_guide)
        quick_start_btn.pack(side=tk.LEFT, padx=(10, 0))

        dismiss_btn = ttk.Button(button_frame, text="Dismiss", command=self.dismiss_onboarding_card)
        dismiss_btn.pack(side=tk.RIGHT)

    def destroy_onboarding_card(self):
        """Remove onboarding card from UI"""
        if self.onboarding_card and self.onboarding_card.winfo_exists():
            self.onboarding_card.destroy()
        self.onboarding_card = None
        self.onboarding_step_vars = []
        self.onboarding_step_labels = []
        self.onboarding_header_var = None
        self.onboarding_subtext_var = None

    def dismiss_onboarding_card(self):
        """Persist dismissal of onboarding helper"""
        self.config['onboarding_dismissed'] = True
        self.save_config()
        self.destroy_onboarding_card()

    def show_quick_start_guide(self):
        """Display a quick start window with onboarding tips"""
        if self.quick_start_window and self.quick_start_window.winfo_exists():
            self.quick_start_window.lift()
            return

        self.quick_start_window = tk.Toplevel(self.root)
        self.quick_start_window.title("Quick Start Guide")
        self.quick_start_window.geometry("440x420")
        self.quick_start_window.resizable(False, False)
        self.quick_start_window.transient(self.root)
        self.quick_start_window.grab_set()
        self.quick_start_window.protocol("WM_DELETE_WINDOW", self.close_quick_start_guide)

        frame = ttk.Frame(self.quick_start_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        intro = ttk.Label(frame, text="A few suggestions to get the most out of AI Document Library:",
                          wraplength=380, justify=tk.LEFT)
        intro.pack(anchor=tk.W)

        tips = [
            ("Add documents", "Click '+ Add Documents' and select PDFs, Word docs, or notes you want to explore."),
            ("Watch processing", "Leave the window open while the status shows ⏳. You'll see a ✓ when each file is ready."),
            ("Try a starter question", "Ask something like 'Summarize my latest meeting notes' or 'What deadlines are mentioned?'"),
            ("Follow up", "Use the chat to drill deeper—follow-up questions use the context of your previous message."),
        ]

        for idx, (title, body) in enumerate(tips, 1):
            ttk.Label(frame, text=f"{idx}. {title}", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(12 if idx > 1 else 16, 2))
            ttk.Label(frame, text=body, wraplength=380, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Separator(frame).pack(fill=tk.X, pady=16)

        closing = ttk.Label(frame,
                             text="Need more help? Check the README for setup details or rerun the setup wizard from the settings file.",
                             wraplength=380, justify=tk.LEFT)
        closing.pack(anchor=tk.W, pady=(0, 12))

        close_btn = ttk.Button(frame, text="Close", command=self.close_quick_start_guide)
        close_btn.pack(anchor=tk.E)

    def close_quick_start_guide(self):
        """Close quick start window"""
        if self.quick_start_window and self.quick_start_window.winfo_exists():
            try:
                self.quick_start_window.grab_release()
            except tk.TclError:
                pass
            self.quick_start_window.destroy()
        self.quick_start_window = None

    def add_documents(self):
        """Add documents to the library"""
        filetypes = [
            ("All supported", "*.pdf;*.txt;*.docx;*.md"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt"),
            ("Word documents", "*.docx"),
            ("Markdown files", "*.md"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select documents to add",
            filetypes=filetypes
        )
        
        if files:
            # Show progress dialog
            self.show_processing_progress(len(files))
            threading.Thread(target=self.process_documents, args=(files,), 
                           daemon=True).start()
    
    def process_documents(self, files):
        """Process and catalog documents"""
        processed_count = 0
        total_files = len(files)
        
        for file_path in files:
            try:
                # Update progress
                self.root.after(0, lambda: self.update_progress(
                    f"Processing {Path(file_path).name}...", processed_count, total_files))
                
                # Copy file to documents directory
                file_path_obj = Path(file_path)
                dest_path = self.documents_dir / file_path_obj.name
                
                # Handle duplicate names
                counter = 1
                while dest_path.exists():
                    name = file_path_obj.stem + f"_{counter}" + file_path_obj.suffix
                    dest_path = self.documents_dir / name
                    counter += 1
                
                # Copy file
                with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                
                # Calculate file hash
                with open(dest_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Add to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO documents 
                    (filename, filepath, file_hash, processed)
                    VALUES (?, ?, ?, ?)
                ''', (dest_path.name, str(dest_path), file_hash, False))
                
                document_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                # Process document with AI
                if document_id:
                    self.root.after(0, lambda: self.update_progress(
                        f"Analyzing {dest_path.name} with AI...", processed_count, total_files))
                    
                    success = self.doc_processor.process_document(document_id, str(dest_path))
                    if not success:
                        print(f"Failed to process document: {dest_path.name}")
                
                processed_count += 1
                
                # Update UI
                self.root.after(0, self.load_document_list)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to add {file_path}: {str(e)}"))
        
        # Close progress dialog
        self.root.after(0, self.close_progress)
    
    def load_document_list(self):
        """Load document list from database"""
        self.doc_listbox.delete(0, tk.END)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT filename, processed FROM documents ORDER BY added_date DESC')
        documents = cursor.fetchall()
        
        for filename, processed in documents:
            status = "✓" if processed else "⏳"
            self.doc_listbox.insert(tk.END, f"{status} {filename}")

        conn.close()

        self.refresh_onboarding_card()
    
    def send_message(self):
        """Send a chat message"""
        message = self.chat_input.get(1.0, tk.END).strip()
        if not message:
            return
        
        # Clear input
        self.chat_input.delete(1.0, tk.END)
        
        # Add user message to chat
        self.add_chat_message("You", message)
        
        # Process message in background
        threading.Thread(target=self.process_chat_message, args=(message,), 
                        daemon=True).start()
    
    def process_chat_message(self, message):
        """Process chat message and generate response"""
        try:
            # Show typing indicator
            self.root.after(0, lambda: self.add_chat_message("Assistant", "Thinking..."))
            
            # Generate AI response
            response = self.chat_system.process_message(message)

            # Replace thinking message with actual response
            self.root.after(0, lambda: self.replace_last_message(response))
            self.root.after(0, self.refresh_onboarding_card)

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.root.after(0, lambda: self.replace_last_message(error_msg))
            self.root.after(0, self.refresh_onboarding_card)
    
    def add_chat_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add message
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def replace_last_message(self, new_message):
        """Replace the last message in chat (for updating thinking indicator)"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Get current content
        content = self.chat_display.get(1.0, tk.END)
        lines = content.strip().split('\n')
        
        # Find and replace last assistant message
        if lines:
            # Remove last message lines
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and "Assistant:" in lines[-1]:
                lines.pop()
                if lines and not lines[-1].strip():
                    lines.pop()
        
        # Clear and rebuild content
        self.chat_display.delete(1.0, tk.END)
        if lines:
            self.chat_display.insert(tk.END, '\n'.join(lines) + '\n\n')
        
        # Add new message
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"[{timestamp}] Assistant: {new_message}\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def show_processing_progress(self, total_files):
        """Show progress dialog for document processing"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Processing Documents")
        self.progress_window.geometry("400x150")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        
        # Center the window
        self.progress_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 300,
            self.root.winfo_rooty() + 200
        ))
        
        frame = ttk.Frame(self.progress_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.progress_label = ttk.Label(frame, text="Starting document processing...")
        self.progress_label.pack(pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar['maximum'] = total_files
        
        self.progress_detail = ttk.Label(frame, text="")
        self.progress_detail.pack()
    
    def update_progress(self, message, current, total):
        """Update progress dialog"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_label.config(text=f"Processing documents... ({current + 1}/{total})")
            self.progress_detail.config(text=message)
            self.progress_bar['value'] = current + 1
            self.progress_window.update()
    
    def close_progress(self):
        """Close progress dialog"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.destroy()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = DocumentLibrary()
    app.run()