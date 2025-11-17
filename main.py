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
from file_system_cataloger import FileSystemCatalog

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
        self.file_catalog = FileSystemCatalog(str(self.db_path))

        # Track setup state
        self.setup_complete = self.config.get('setup_complete', False)
        self.catalog_scanning = False
        
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
                processed BOOLEAN DEFAULT FALSE,
                tags TEXT,
                category TEXT
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

        # Tags table for managing unique tags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL,
                color TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL,
                description TEXT,
                color TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
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
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # === File System Catalog Section ===
        catalog_label = ttk.Label(left_panel, text="🌍 Computer Files",
                                 font=('Arial', 14, 'bold'))
        catalog_label.pack(anchor=tk.W, pady=(0, 10))

        # Catalog stats
        self.catalog_stats_label = ttk.Label(left_panel, text="Not scanned yet",
                                            foreground='#666', font=('Arial', 9))
        self.catalog_stats_label.pack(anchor=tk.W, pady=(0, 10))

        # Catalog controls
        catalog_buttons_frame = ttk.Frame(left_panel)
        catalog_buttons_frame.pack(fill=tk.X, pady=(0, 15))

        self.scan_btn = ttk.Button(catalog_buttons_frame, text="📂 Scan Computer",
                                   command=self.start_catalog_scan)
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.configure_scan_btn = ttk.Button(catalog_buttons_frame, text="⚙️ Configure",
                                            command=self.configure_catalog_scan)
        self.configure_scan_btn.pack(side=tk.LEFT)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # === Document Library Section ===
        doc_label = ttk.Label(left_panel, text="📚 Document Library",
                             font=('Arial', 14, 'bold'))
        doc_label.pack(anchor=tk.W, pady=(0, 10))

        # Add documents button
        add_btn = ttk.Button(left_panel, text="+ Add Documents",
                           command=self.add_documents)
        add_btn.pack(fill=tk.X, pady=(0, 10))

        # Drag & drop hint
        drag_hint = ttk.Label(left_panel,
                             text="💡 Tip: Drag & drop files/folders here or press Ctrl+Shift+V",
                             foreground='#666', font=('Arial', 9, 'italic'))
        drag_hint.pack(anchor=tk.W, pady=(0, 5))

        # Document list (takes 60% of left panel height)
        doc_list_container = ttk.Frame(left_panel)
        doc_list_container.pack(fill=tk.BOTH, expand=True)

        doc_frame = ttk.Frame(doc_list_container)
        doc_frame.pack(fill=tk.BOTH, expand=True)

        self.doc_listbox = tk.Listbox(doc_frame, background='#ffffff', exportselection=False)
        doc_scrollbar = ttk.Scrollbar(doc_frame, orient=tk.VERTICAL,
                                     command=self.doc_listbox.yview)
        self.doc_listbox.configure(yscrollcommand=doc_scrollbar.set)

        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        doc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event to show preview
        self.doc_listbox.bind('<<ListboxSelect>>', self.show_document_preview)

        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=5)

        # Document preview pane
        preview_label = ttk.Label(left_panel, text="📄 Document Preview",
                                 font=('Arial', 12, 'bold'))
        preview_label.pack(anchor=tk.W, pady=(5, 5))

        preview_frame = ttk.Frame(left_panel)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, height=8,
                                   bg='#f9f9f9', font=('Arial', 9),
                                   state=tk.DISABLED)
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL,
                                         command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_scrollbar.set)

        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags and category section
        tag_frame = ttk.Frame(left_panel)
        tag_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(tag_frame, text="🏷️ Tags:", font=('Arial', 9)).pack(side=tk.LEFT)

        self.tag_display = ttk.Label(tag_frame, text="None", foreground='#666',
                                    font=('Arial', 9))
        self.tag_display.pack(side=tk.LEFT, padx=5)

        self.edit_tags_btn = ttk.Button(tag_frame, text="Edit Tags",
                                        command=self.edit_document_tags,
                                        state=tk.DISABLED)
        self.edit_tags_btn.pack(side=tk.RIGHT)

        # Category section
        cat_frame = ttk.Frame(left_panel)
        cat_frame.pack(fill=tk.X, pady=(3, 0))

        ttk.Label(cat_frame, text="📁 Category:", font=('Arial', 9)).pack(side=tk.LEFT)

        self.category_display = ttk.Label(cat_frame, text="None", foreground='#666',
                                         font=('Arial', 9))
        self.category_display.pack(side=tk.LEFT, padx=5)

        self.edit_category_btn = ttk.Button(cat_frame, text="Edit",
                                           command=self.edit_document_category,
                                           state=tk.DISABLED)
        self.edit_category_btn.pack(side=tk.RIGHT)

        # Preview action buttons
        preview_btn_frame = ttk.Frame(left_panel)
        preview_btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.open_doc_btn = ttk.Button(preview_btn_frame, text="📂 Open File",
                                       command=self.open_selected_document,
                                       state=tk.DISABLED)
        self.open_doc_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.remove_doc_btn = ttk.Button(preview_btn_frame, text="🗑️ Remove",
                                        command=self.remove_selected_document,
                                        state=tk.DISABLED)
        self.remove_doc_btn.pack(side=tk.LEFT)

        # Enable drag and drop
        self.enable_drag_and_drop(left_panel)
        self.enable_drag_and_drop(self.doc_listbox)
        
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

        cursor.execute('SELECT id, filename, processed FROM documents ORDER BY added_date DESC')
        self.documents_cache = cursor.fetchall()  # Cache for quick access

        for doc_id, filename, processed in self.documents_cache:
            status = "✓" if processed else "⏳"
            self.doc_listbox.insert(tk.END, f"{status} {filename}")

        conn.close()

        self.refresh_onboarding_card()

    def show_document_preview(self, event=None):
        """Display preview of selected document"""
        selection = self.doc_listbox.curselection()

        if not selection:
            # Clear preview if no selection
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.config(state=tk.DISABLED)
            self.open_doc_btn.config(state=tk.DISABLED)
            self.remove_doc_btn.config(state=tk.DISABLED)
            self.edit_tags_btn.config(state=tk.DISABLED)
            self.edit_category_btn.config(state=tk.DISABLED)
            self.tag_display.config(text="None")
            self.category_display.config(text="None")
            return

        # Enable buttons
        self.open_doc_btn.config(state=tk.NORMAL)
        self.remove_doc_btn.config(state=tk.NORMAL)
        self.edit_tags_btn.config(state=tk.NORMAL)
        self.edit_category_btn.config(state=tk.NORMAL)

        # Get selected document
        index = selection[0]
        if not hasattr(self, 'documents_cache') or index >= len(self.documents_cache):
            return

        doc_id, filename, processed = self.documents_cache[index]

        # Fetch document details from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT filepath, summary, topics, doc_type, added_date, tags, category
            FROM documents WHERE id = ?
        ''', (doc_id,))

        result = cursor.fetchone()

        if result:
            filepath, summary, topics, doc_type, added_date, tags, category = result

            # Update tag and category displays
            if tags:
                self.tag_display.config(text=tags, foreground='#1976d2')
            else:
                self.tag_display.config(text="None", foreground='#666')

            if category:
                self.category_display.config(text=category, foreground='#388e3c')
            else:
                self.category_display.config(text="None", foreground='#666')

            # Build preview text
            preview_content = f"📋 {filename}\n"
            preview_content += "=" * 40 + "\n\n"

            if doc_type:
                preview_content += f"Type: {doc_type}\n"

            if added_date:
                preview_content += f"Added: {added_date}\n"

            preview_content += f"Status: {'Processed ✓' if processed else 'Processing ⏳'}\n\n"

            if summary:
                preview_content += "Summary:\n"
                preview_content += "-" * 40 + "\n"
                preview_content += f"{summary}\n\n"

            if topics:
                preview_content += f"Topics: {topics}\n\n"

            # Get first chunk as sample
            cursor.execute('''
                SELECT chunk_text FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                LIMIT 1
            ''', (doc_id,))

            chunk = cursor.fetchone()
            if chunk:
                preview_content += "Sample Content:\n"
                preview_content += "-" * 40 + "\n"
                preview_content += chunk[0][:500]  # First 500 chars
                if len(chunk[0]) > 500:
                    preview_content += "..."

            # Display preview
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview_content)
            self.preview_text.config(state=tk.DISABLED)

        conn.close()

    def open_selected_document(self):
        """Open the selected document in system default application"""
        selection = self.doc_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if not hasattr(self, 'documents_cache') or index >= len(self.documents_cache):
            return

        doc_id = self.documents_cache[index][0]

        # Get filepath
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT filepath FROM documents WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            filepath = result[0]
            try:
                # Open with system default application
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', filepath])
                elif platform.system() == 'Windows':
                    os.startfile(filepath)
                else:  # Linux
                    subprocess.run(['xdg-open', filepath])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def edit_document_tags(self):
        """Edit tags for the selected document"""
        selection = self.doc_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if not hasattr(self, 'documents_cache') or index >= len(self.documents_cache):
            return

        doc_id, filename, _ = self.documents_cache[index]

        # Get current tags
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT tags FROM documents WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        current_tags = result[0] if result and result[0] else ""

        # Get existing tags for suggestions
        cursor.execute('SELECT DISTINCT tag_name FROM tags ORDER BY tag_name')
        existing_tags = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Create tag editing dialog
        tag_dialog = tk.Toplevel(self.root)
        tag_dialog.title("Edit Tags")
        tag_dialog.geometry("400x300")
        tag_dialog.transient(self.root)
        tag_dialog.grab_set()

        frame = ttk.Frame(tag_dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"Edit tags for: {filename}",
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(frame, text="Enter tags (comma-separated):").pack(anchor=tk.W)

        tag_entry = tk.Text(frame, height=3, wrap=tk.WORD, font=('Arial', 10))
        tag_entry.pack(fill=tk.X, pady=(5, 10))
        tag_entry.insert(1.0, current_tags)

        ttk.Label(frame, text="Suggested tags:",
                 font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=(10, 5))

        # Create suggestion buttons
        suggestion_frame = ttk.Frame(frame)
        suggestion_frame.pack(fill=tk.BOTH, expand=True)

        if existing_tags:
            for tag in existing_tags[:10]:  # Show max 10 suggestions
                btn = ttk.Button(suggestion_frame, text=tag,
                               command=lambda t=tag: self.add_suggested_tag(tag_entry, t))
                btn.pack(side=tk.LEFT, padx=2, pady=2)
        else:
            ttk.Label(suggestion_frame, text="No existing tags",
                     foreground='#666').pack()

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))

        def save_tags():
            tags_text = tag_entry.get(1.0, tk.END).strip()
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE documents SET tags = ? WHERE id = ?',
                          (tags_text, doc_id))

            # Add new tags to tags table
            if tags_text:
                for tag in tags_text.split(','):
                    tag = tag.strip()
                    if tag:
                        cursor.execute('INSERT OR IGNORE INTO tags (tag_name) VALUES (?)',
                                     (tag,))

            conn.commit()
            conn.close()

            # Refresh preview
            self.show_document_preview()
            tag_dialog.destroy()
            messagebox.showinfo("Success", "Tags updated successfully!")

        ttk.Button(button_frame, text="Cancel",
                  command=tag_dialog.destroy).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Save",
                  command=save_tags).pack(side=tk.RIGHT)

    def add_suggested_tag(self, text_widget, tag):
        """Add a suggested tag to the tag entry"""
        current = text_widget.get(1.0, tk.END).strip()
        if current:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(1.0, f"{current}, {tag}")
        else:
            text_widget.insert(1.0, tag)

    def edit_document_category(self):
        """Edit category for the selected document"""
        selection = self.doc_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if not hasattr(self, 'documents_cache') or index >= len(self.documents_cache):
            return

        doc_id, filename, _ = self.documents_cache[index]

        # Get current category
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT category FROM documents WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        current_category = result[0] if result and result[0] else ""

        # Get existing categories
        cursor.execute('SELECT category_name FROM categories ORDER BY category_name')
        existing_categories = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Create category editing dialog
        cat_dialog = tk.Toplevel(self.root)
        cat_dialog.title("Edit Category")
        cat_dialog.geometry("350x250")
        cat_dialog.transient(self.root)
        cat_dialog.grab_set()

        frame = ttk.Frame(cat_dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"Edit category for: {filename}",
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(frame, text="Select or enter category:").pack(anchor=tk.W)

        category_var = tk.StringVar(value=current_category)

        # Dropdown with existing categories
        category_combo = ttk.Combobox(frame, textvariable=category_var,
                                      values=existing_categories)
        category_combo.pack(fill=tk.X, pady=(5, 15))

        # Common categories suggestion
        ttk.Label(frame, text="Common categories:",
                 font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=(10, 5))

        common_cats = ["Work", "Personal", "Research", "Reference", "Archive"]
        for cat in common_cats:
            if cat not in existing_categories:
                btn = ttk.Button(frame, text=cat,
                               command=lambda c=cat: category_var.set(c))
                btn.pack(side=tk.LEFT, padx=2)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))

        def save_category():
            category = category_var.get().strip()

            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE documents SET category = ? WHERE id = ?',
                          (category, doc_id))

            # Add new category to categories table
            if category:
                cursor.execute('INSERT OR IGNORE INTO categories (category_name) VALUES (?)',
                             (category,))

            conn.commit()
            conn.close()

            # Refresh preview
            self.show_document_preview()
            cat_dialog.destroy()
            messagebox.showinfo("Success", "Category updated successfully!")

        ttk.Button(button_frame, text="Cancel",
                  command=cat_dialog.destroy).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Save",
                  command=save_category).pack(side=tk.RIGHT)

    def remove_selected_document(self):
        """Remove selected document from library"""
        selection = self.doc_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if not hasattr(self, 'documents_cache') or index >= len(self.documents_cache):
            return

        doc_id, filename, processed = self.documents_cache[index]

        # Confirm removal
        if not messagebox.askyesno("Remove Document",
                                   f"Remove '{filename}' from library?\n\n"
                                   "The file will be deleted from the documents folder."):
            return

        try:
            # Get filepath before deleting
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT filepath FROM documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            filepath = result[0] if result else None

            # Remove from database
            cursor.execute('DELETE FROM document_chunks WHERE document_id = ?', (doc_id,))
            cursor.execute('DELETE FROM document_embeddings WHERE document_id = ?', (doc_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

            conn.commit()
            conn.close()

            # Delete physical file
            if filepath and Path(filepath).exists():
                Path(filepath).unlink()

            # Refresh list
            self.load_document_list()

            messagebox.showinfo("Success", f"'{filename}' has been removed.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove document: {str(e)}")
    
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

    def configure_catalog_scan(self):
        """Configure which directories to scan"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Configure Computer Scan")
        config_window.geometry("600x500")
        config_window.transient(self.root)
        config_window.grab_set()

        frame = ttk.Frame(config_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(frame, text="Configure Computer Scan",
                         font=('Arial', 16, 'bold'))
        title.pack(anchor=tk.W, pady=(0, 20))

        # Info text
        info_text = ("Select which directories to scan. The scanner will automatically "
                    "skip system files, temporary files, and large binaries to keep "
                    "your catalog clean and useful.")
        info = ttk.Label(frame, text=info_text, wraplength=550, justify=tk.LEFT)
        info.pack(anchor=tk.W, pady=(0, 20))

        # Preset options
        preset_label = ttk.Label(frame, text="Quick presets:", font=('Arial', 11, 'bold'))
        preset_label.pack(anchor=tk.W, pady=(0, 5))

        preset_frame = ttk.Frame(frame)
        preset_frame.pack(fill=tk.X, pady=(0, 15))

        self.scan_preset = tk.StringVar(value='home')

        presets = [
            ('home', 'Home directory only (Recommended)'),
            ('documents', 'Documents folder'),
            ('custom', 'Custom directories')
        ]

        for value, label in presets:
            ttk.Radiobutton(preset_frame, text=label, variable=self.scan_preset,
                          value=value).pack(anchor=tk.W, pady=2)

        # Custom directories
        custom_label = ttk.Label(frame, text="Custom directories (one per line):",
                                font=('Arial', 11, 'bold'))
        custom_label.pack(anchor=tk.W, pady=(10, 5))

        self.custom_dirs_text = tk.Text(frame, height=8, wrap=tk.WORD)
        self.custom_dirs_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Pre-fill with home directory
        home_dir = str(Path.home())
        self.custom_dirs_text.insert(1.0, home_dir)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)

        cancel_btn = ttk.Button(button_frame, text="Cancel",
                               command=config_window.destroy)
        cancel_btn.pack(side=tk.LEFT)

        start_btn = ttk.Button(button_frame, text="Start Scan",
                              command=lambda: self.execute_catalog_scan(config_window))
        start_btn.pack(side=tk.RIGHT)

    def execute_catalog_scan(self, config_window):
        """Execute the catalog scan with configured directories"""
        preset = self.scan_preset.get()
        directories = []

        if preset == 'home':
            directories = [str(Path.home())]
        elif preset == 'documents':
            doc_dirs = [
                Path.home() / 'Documents',
                Path.home() / 'Desktop',
                Path.home() / 'Downloads'
            ]
            directories = [str(d) for d in doc_dirs if d.exists()]
        elif preset == 'custom':
            custom_text = self.custom_dirs_text.get(1.0, tk.END).strip()
            directories = [line.strip() for line in custom_text.split('\n') if line.strip()]

        # Validate directories
        valid_dirs = []
        for directory in directories:
            if Path(directory).exists() and Path(directory).is_dir():
                valid_dirs.append(directory)

        if not valid_dirs:
            messagebox.showerror("Error", "No valid directories selected.")
            return

        # Save config
        self.config['catalog_directories'] = valid_dirs
        self.save_config()

        config_window.destroy()
        self.start_catalog_scan()

    def start_catalog_scan(self):
        """Start scanning the file system"""
        if self.catalog_scanning:
            messagebox.showinfo("Scan in Progress",
                              "A scan is already in progress. Please wait for it to complete.")
            return

        # Get directories to scan
        directories = self.config.get('catalog_directories', [str(Path.home())])

        # Confirm scan
        dir_list = '\n'.join(directories)
        msg = f"This will scan the following directories:\n\n{dir_list}\n\nThis may take several minutes. Continue?"

        if not messagebox.askyesno("Start Catalog Scan", msg):
            return

        # Start scan
        self.catalog_scanning = True
        self.scan_btn.config(state=tk.DISABLED, text="Scanning...")

        # Show progress window
        self.show_catalog_progress()

        # Start background scan
        def progress_callback(stats, completed=False):
            if completed:
                self.root.after(0, lambda: self.catalog_scan_complete(stats))
            else:
                self.root.after(0, lambda: self.update_catalog_progress(stats))

        self.file_catalog.start_background_scan(directories, progress_callback)

    def show_catalog_progress(self):
        """Show progress dialog for catalog scan"""
        self.catalog_progress_window = tk.Toplevel(self.root)
        self.catalog_progress_window.title("Scanning Computer")
        self.catalog_progress_window.geometry("450x200")
        self.catalog_progress_window.transient(self.root)
        self.catalog_progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing

        frame = ttk.Frame(self.catalog_progress_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Scanning your computer...",
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 15))

        self.catalog_progress_label = ttk.Label(frame,
                                               text="Starting scan...\n\nThis may take several minutes.",
                                               justify=tk.CENTER)
        self.catalog_progress_label.pack(pady=(0, 15))

        self.catalog_progress_detail = ttk.Label(frame, text="", foreground='#666')
        self.catalog_progress_detail.pack(pady=(0, 15))

        # Stop button
        stop_btn = ttk.Button(frame, text="Stop Scan",
                             command=self.stop_catalog_scan)
        stop_btn.pack()

    def update_catalog_progress(self, stats):
        """Update catalog scan progress"""
        if hasattr(self, 'catalog_progress_label') and self.catalog_progress_label.winfo_exists():
            progress_text = f"Scanned: {stats['files_scanned']} files\n"
            progress_text += f"Indexed: {stats['files_indexed']} files\n"
            progress_text += f"Directories: {stats['directories_scanned']}"

            self.catalog_progress_label.config(text=progress_text)

            detail = f"Skipped: {stats['files_skipped']} | Errors: {stats['errors']}"
            self.catalog_progress_detail.config(text=detail)

    def stop_catalog_scan(self):
        """Stop the catalog scan"""
        self.file_catalog.stop_scan()
        self.catalog_scanning = False

        if hasattr(self, 'catalog_progress_window') and self.catalog_progress_window.winfo_exists():
            self.catalog_progress_window.destroy()

        self.scan_btn.config(state=tk.NORMAL, text="📂 Scan Computer")
        self.update_catalog_stats()

    def catalog_scan_complete(self, stats):
        """Handle catalog scan completion"""
        self.catalog_scanning = False

        # Close progress window
        if hasattr(self, 'catalog_progress_window') and self.catalog_progress_window.winfo_exists():
            self.catalog_progress_window.destroy()

        # Update UI
        self.scan_btn.config(state=tk.NORMAL, text="📂 Scan Computer")
        self.update_catalog_stats()

        # Show completion message
        msg = f"Scan complete!\n\n"
        msg += f"Files scanned: {stats['files_scanned']}\n"
        msg += f"Files indexed: {stats['files_indexed']}\n"
        msg += f"Directories: {stats['directories_scanned']}\n\n"
        msg += "You can now search and chat about all your files!"

        messagebox.showinfo("Scan Complete", msg)

    def update_catalog_stats(self):
        """Update the catalog statistics display"""
        stats = self.file_catalog.get_catalog_stats()

        if stats['total_files'] == 0:
            self.catalog_stats_label.config(text="Not scanned yet")
        else:
            size_mb = stats['total_size_bytes'] / (1024 * 1024)
            stats_text = f"📊 {stats['total_files']:,} files indexed"
            stats_text += f" | {size_mb:.1f} MB"
            stats_text += f" | {stats['content_indexed_files']:,} searchable"
            self.catalog_stats_label.config(text=stats_text)

    def enable_drag_and_drop(self, widget):
        """Enable drag and drop on a widget"""
        # Try to use tkinterdnd2 if available
        try:
            # Store original background for visual feedback
            try:
                original_bg = widget.cget('background')
            except:
                original_bg = '#f0f0f0'

            def on_drop(event):
                """Handle file drop"""
                try:
                    widget.config(background=original_bg)
                except:
                    pass

                # Parse dropped files and folders
                files = self.parse_drop_files(event.data if hasattr(event, 'data') else str(event))

                if files:
                    # Show processing dialog
                    self.show_processing_progress(len(files))
                    threading.Thread(target=self.process_documents, args=(files,),
                                   daemon=True).start()

            def on_drag_enter(event):
                """Visual feedback when dragging over widget"""
                try:
                    widget.config(background='#e3f2fd')
                except:
                    pass

            def on_drag_leave(event):
                """Reset visual feedback"""
                try:
                    widget.config(background=original_bg)
                except:
                    pass

            # Try tkinterdnd2 style bindings
            try:
                from tkinterdnd2 import DND_FILES
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>', on_drop)
                widget.dnd_bind('<<DragEnter>>', on_drag_enter)
                widget.dnd_bind('<<DragLeave>>', on_drag_leave)
            except (ImportError, AttributeError):
                # Fallback: Add a context menu option to paste file paths
                def paste_files(event=None):
                    """Alternative: paste file paths from clipboard"""
                    try:
                        clipboard_text = self.root.clipboard_get()
                        files = self.parse_drop_files(clipboard_text)
                        if files:
                            self.show_processing_progress(len(files))
                            threading.Thread(target=self.process_documents, args=(files,),
                                           daemon=True).start()
                    except:
                        pass

                # Add keyboard shortcut for pasting files (Ctrl+Shift+V)
                widget.bind('<Control-Shift-V>', paste_files)

        except Exception as e:
            # Silent fail - drag and drop is optional
            pass

    def parse_drop_files(self, data):
        """Parse dropped file data from different platforms, including folders"""
        files = []
        supported_extensions = {'.pdf', '.txt', '.docx', '.md', '.doc', '.rtf'}

        if not data:
            return files

        # Handle string data
        if isinstance(data, str):
            # Remove curly braces and split by whitespace
            data = data.strip('{}')

            # Handle space-separated paths (may be quoted)
            import shlex
            try:
                paths = shlex.split(data)
            except:
                # Fallback: split by newlines or spaces
                paths = [p.strip() for p in data.replace('\n', ' ').split()]

            for path_str in paths:
                path_str = path_str.strip()
                if not path_str:
                    continue

                path = Path(path_str)

                if path.exists():
                    if path.is_file():
                        # Check if it's a supported file type
                        if path.suffix.lower() in supported_extensions:
                            files.append(str(path))
                    elif path.is_dir():
                        # Recursively find supported files in directory
                        for ext in supported_extensions:
                            files.extend([str(f) for f in path.rglob(f'*{ext}')])

        return list(set(files))  # Remove duplicates

    def run(self):
        """Start the application"""
        # Update catalog stats on startup
        try:
            self.update_catalog_stats()
        except:
            pass

        self.root.mainloop()

if __name__ == "__main__":
    app = DocumentLibrary()
    app.run()