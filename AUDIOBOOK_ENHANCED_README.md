# 🎧 Chatterbox Audiobook Studio - Enhanced Edition

**The ultimate audiobook creation toolkit combining advanced TTS with professional audiobook features.**

## 🚀 What's New

This enhanced edition merges all audiobook features from the `chatterbox-Audiobook` project into the main Chatterbox TTS, providing:

### ✨ **Enhanced Features**
- **📚 Voice Library System**: Organize and manage character voices
- **🎯 Long Text Processing**: Handle books of any size with intelligent chunking
- **🎭 Multi-Voice Support**: Create dynamic audiobooks with character voice mapping
- **🔇 Return Pause System**: Natural speech flow with automatic pauses
- **📋 Batch Processing**: Process multiple chapters sequentially
- **🎚️ Professional Volume Control**: Consistent audio levels across all voices

## 🎯 Quick Start

### 1. Launch the Enhanced App
```bash
python gradio_audiobook_app.py
```

### 2. Open Your Browser
The app will automatically open at `http://localhost:7860`

## 📋 Available Tabs

### 🎤 Basic TTS
- Single text synthesis with voice profiles
- Multi-language support (23 languages)
- Real-time parameter adjustment

### 📚 Audiobook Creation
- **Single Voice Mode**: Generate entire books with one consistent voice
- **Text Chunking**: Automatic splitting for long texts
- **File Upload**: Support for .txt and .md files
- **Project Management**: Organized output with project names

### 📚 Voice Library
- **Voice Profiles**: Save custom voices with settings
- **Reference Audio**: Upload 10-30 second samples
- **Parameter Storage**: Store exaggeration, CFG, and temperature settings
- **Easy Management**: Create, load, and delete voice profiles

### 🎭 Multi-Voice Creation
- **Character Mapping**: Assign voices to story characters
- **Dialogue Parsing**: Automatic parsing of [Character] dialogue format
- **Natural Pauses**: Line breaks create natural speech timing
- **Batch Generation**: Process entire books with character switching

## 🎭 Text Formatting Guide

### Single Voice Format
```
Chapter 1: The Beginning

It was a dark and stormy night. The wind howled through the trees,
creating an eerie atmosphere that made everyone uneasy.

The old house stood at the top of the hill, its windows dark and
foreboding against the stormy sky.
```

### Multi-Voice Format
```
[Narrator] The detective entered the room cautiously.

[Detective] "Something doesn't feel right about this case."

[Suspect] "I told you everything I know!"

[Detective] "Then why does your story keep changing?"

[Narrator] The tension in the room was palpable as both parties stared at each other.
```

### Advanced Formatting Tips
- **Line Breaks**: Each line break adds 0.1 seconds of pause
- **Scene Breaks**: Use double line breaks for major transitions
- **Character Consistency**: Always use the same character names
- **Emphasis**: Add extra line breaks before important moments

## 🎤 Voice Profile Creation

### Step 1: Prepare Reference Audio
- **Duration**: 10-30 seconds optimal
- **Quality**: Clear, noise-free recording
- **Content**: Natural speech, not overly dramatic
- **Format**: WAV, MP3, or FLAC accepted

### Step 2: Configure Settings
- **Exaggeration**: 0.3-0.7 for most voices
- **CFG Weight**: 0.3-0.8 (lower = slower, more deliberate)
- **Temperature**: 0.6-1.2 (higher = more variation)

### Step 3: Save and Organize
- **Naming**: Use descriptive names like "narrator_deep" or "character_young_female"
- **Descriptions**: Include character notes and usage guidelines
- **Backup**: Keep your voice_library folder backed up

## 🎯 Professional Workflows

### Audiobook Production Pipeline
1. **Pre-production**: Plan character voices and gather reference audio
2. **Voice Creation**: Create voice profiles for each character
3. **Text Preparation**: Format your manuscript with proper dialogue tags
4. **Generation**: Use appropriate tab for single or multi-voice creation
5. **Quality Control**: Review and regenerate sections as needed
6. **Export**: Download final audiobook files

### Batch Processing
- **Chapter-by-Chapter**: Process large books in manageable chunks
- **Queue Management**: Upload multiple text files for sequential processing
- **Progress Monitoring**: Real-time updates on generation progress

## 🔧 Technical Specifications

### Supported Formats
- **Input**: TXT, MD files
- **Audio Input**: WAV, MP3, FLAC for voice cloning
- **Output**: High-quality WAV files
- **Languages**: 23 supported languages

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU recommended
- **Storage**: Sufficient space for voice library and generated audio

### Voice Library Structure
```
voice_library/
├── narrator_male/
│   ├── config.json
│   └── reference.wav
├── character_female/
│   ├── config.json
│   └── reference.mp3
└── villain_voice/
    ├── config.json
    └── reference.wav
```

## 🎨 Best Practices

### Voice Creation
- **Consistency**: Use the same voice profile for each character throughout
- **Testing**: Test voices with sample text before full production
- **Documentation**: Keep notes on character voice characteristics
- **Backup**: Regular backups of your voice library

### Text Preparation
- **Clean Formatting**: Remove unnecessary formatting
- **Character Tags**: Use consistent character naming
- **Natural Breaks**: Add line breaks for natural speech timing
- **Proofreading**: Ensure text is clean and error-free

### Quality Optimization
- **Reference Quality**: Use high-quality reference audio
- **Parameter Tuning**: Adjust settings based on character personality
- **Segment Testing**: Test small segments before full generation
- **Volume Consistency**: Ensure consistent levels across all voices

## 🚀 Advanced Features

### Return Pause System
- **Automatic Detection**: Counts line breaks automatically
- **Configurable Timing**: 0.1 seconds per line break
- **Accumulative Pauses**: Multiple breaks create longer pauses
- **Real-time Feedback**: Console shows pause calculations

### Memory Management
- **Chunk Processing**: Handles large texts without memory issues
- **Background Processing**: Non-blocking generation for large projects
- **Progress Tracking**: Real-time progress updates
- **Error Recovery**: Automatic retry on generation failures

## 🆘 Troubleshooting

### Common Issues
- **CUDA Errors**: Use CPU mode for multi-voice generation
- **Memory Issues**: Enable chunking for large texts
- **Voice Quality**: Check reference audio quality and settings
- **Format Issues**: Ensure proper text formatting

### Performance Tips
- **GPU Usage**: Use GPU for single voice, CPU for multi-voice
- **Batch Size**: Process large books in smaller chunks
- **Voice Library**: Keep voice profiles organized
- **Monitoring**: Watch console for progress updates

## 📚 Examples and Templates

### Voice Profile Examples
```json
{
  "voice_name": "narrator_professional",
  "display_name": "Professional Narrator",
  "description": "Clear, neutral voice for narration",
  "exaggeration": 0.4,
  "cfg_weight": 0.5,
  "temperature": 0.7
}
```

### Character Voice Settings
- **Hero**: Exaggeration 0.6, CFG 0.6, Temp 0.8
- **Villain**: Exaggeration 0.8, CFG 0.4, Temp 0.9
- **Narrator**: Exaggeration 0.3, CFG 0.5, Temp 0.6
- **Comic Relief**: Exaggeration 0.9, CFG 0.7, Temp 1.1

## 🎯 Next Steps

1. **Launch the App**: Run `python gradio_audiobook_app.py`
2. **Create Voice Profiles**: Set up your character voices
3. **Prepare Your Text**: Format your manuscript
4. **Generate Your Audiobook**: Use the appropriate creation tab
5. **Refine and Export**: Review and download your final audiobook

---

**Ready to create professional audiobooks? Launch the app and start building your voice library! 🎧✨**
