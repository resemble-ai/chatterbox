# 🎧 Chatterbox Audiobook Studio - Enhanced Edition

**The ultimate open-source audiobook creation toolkit with 24-language TTS, voice cloning, and multi-character support.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-orange.svg)](https://gradio.app)
[![GitHub](https://img.shields.io/badge/GitHub-AmiXDme-blue)](https://github.com/AmiXDme/chatterbox)

## ✨ What's New in Enhanced Edition

### 🎭 **Complete Audiobook Creation Suite**
- **24 Language Support** including **Bangla (bn)** ✅
- **Voice Library Management** - Create and manage character voices
- **Long Text Processing** - Handle books up to 100,000+ words
- **Multi-Voice Support** - Character dialogue with voice mapping
- **Public URL Sharing** - Share your interface worldwide
- **Professional Audio** - High-quality TTS with parameter control
- **One-Click Setup** - Windows batch file included

## 🚀 **Quick Start - 3 Ways**

### **🎯 Method 1: One-Click Setup (Recommended)**
```bash
# Windows: Double-click run_webui.bat
# Or run in terminal:
python gradio_audiobook_app.py
```

### **🎯 Method 2: Manual Setup**
```bash
# Clone the repository
git clone https://github.com/AmiXDme/chatterbox.git
cd chatterbox

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the interface
python gradio_audiobook_app.py
```

### **🎯 Method 3: Public URL**
```bash
# Automatically generates shareable URL
python gradio_audiobook_app.py --share
```

## 🌍 **Supported Languages (24 Total)**

| Language | Code | Status |
|----------|------|--------|
| **Bangla** | **bn** | ✅ **NEW** |
| Arabic | ar | ✅ |
| Chinese | zh | ✅ |
| Danish | da | ✅ |
| English | en | ✅ |
| Finnish | fi | ✅ |
| French | fr | ✅ |
| German | de | ✅ |
| Greek | el | ✅ |
| Hebrew | he | ✅ |
| Hindi | hi | ✅ |
| Italian | it | ✅ |
| Japanese | ja | ✅ |
| Korean | ko | ✅ |
| Malay | ms | ✅ |
| Dutch | nl | ✅ |
| Norwegian | no | ✅ |
| Polish | pl | ✅ |
| Portuguese | pt | ✅ |
| Russian | ru | ✅ |
| Spanish | es | ✅ |
| Swedish | sv | ✅ |
| Swahili | sw | ✅ |
| Turkish | tr | ✅ |

## 📋 **Usage Guide**

### **🎤 Basic TTS**
1. **Enter text** in any of 24 supported languages
2. **Select voice profile** (optional)
3. **Adjust parameters** (exaggeration, CFG, temperature)
4. **Generate speech** instantly

### **📚 Audiobook Creation**
1. **Upload text file** (.txt, .md) or paste content
2. **Choose project name**
3. **Select voice profile**
4. **Enable chunking** for long texts
5. **Generate complete audiobook**

### **🎭 Multi-Voice Creation**
1. **Format text** with character tags:
   ```
   [Narrator] The story begins...
   
   [Character1] Hello there!
   
   [Character2] How are you?
   ```
2. **Map characters to voices**
3. **Generate multi-voice audiobook**

### **📚 Voice Library**
1. **Upload reference audio** (10-30 seconds)
2. **Configure voice settings**
3. **Save voice profile**
4. **Use across all projects**

## 🛠️ **Installation & Setup**

### **System Requirements**
- **Python 3.10+**
- **8GB+ RAM** (16GB recommended for large texts)
- **Modern browser** for web interface
- **Optional:** CUDA-compatible GPU for faster processing

### **Quick Install Commands**
```bash
# Windows (recommended)
run_webui.bat

# Linux/macOS
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .
python gradio_audiobook_app.py
```

## 🖥️ **Interface Overview**

### **Tab 1: 🎤 Basic TTS**
- Single text synthesis
- Real-time parameter adjustment
- Multi-language support

### **Tab 2: 📚 Audiobook Creation**
- Long text processing
- File upload support
- Project management
- Chunking for large texts

### **Tab 3: 📚 Voice Library**
- Voice profile creation
- Reference audio upload
- Voice management
- Settings storage

### **Tab 4: 🎭 Multi-Voice**
- Character dialogue parsing
- Voice mapping
- Natural pause insertion
- Batch processing

## 📁 **Project Structure**

```
chatterbox/
├── 📁 src/chatterbox/          # Core TTS modules
├── 📁 voice_library/           # Voice profiles
├── 📁 audiobook_projects/      # Generated audiobooks
├── 📁 models/                  # Downloaded TTS models
├── 📁 venv/                    # Virtual environment
├── 📄 requirements.txt         # Dependencies
├── 📄 run_webui.bat           # Windows launcher
├── 📄 gradio_audiobook_app.py # Main interface
├── 📄 interface_preview.html  # UI preview
├── 📄 README.md               # This file
└── 📄 LICENSE                 # MIT license
```

## 🎯 **Quick Commands**

```bash
# Start with public URL
python gradio_audiobook_app.py --share

# Start locally only
python gradio_audiobook_app.py

# Windows one-click
run_webui.bat
```

## 🐛 **Troubleshooting**

### **Common Issues & Solutions**
- **CUDA errors:** Use CPU mode automatically
- **Memory issues:** Enable chunking for large texts
- **Model download:** First run takes 5-10 minutes
- **Port conflicts:** Uses 7860 by default

### **Performance Tips**
- **GPU:** Use CUDA for faster processing
- **Memory:** Process large books in chunks
- **Voice quality:** Use 10-30 second reference audio
- **Batch processing:** Upload multiple files

## 🔗 **GitHub Upload Commands**

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Enhanced audiobook studio with 24-language support"

# Add remote and push
git remote add origin https://github.com/AmiXDme/chatterbox.git
git branch -M main
git push -u origin main
```

## 🌟 **Get Started Now**

```bash
# Clone and run
git clone https://github.com/AmiXDme/chatterbox.git
cd chatterbox
run_webui.bat
```

**Visit:** `http://localhost:7860` or use the generated **public URL** to share worldwide!

---

**🎧 Ready to create professional audiobooks in 24 languages? Start now!**
