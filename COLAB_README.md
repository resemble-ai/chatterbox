# 🎙️ Chatterbox TTS - Google Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/viveksurmay/chatterbox-colab/blob/master/chatterbox_colab.ipynb)

This repository contains a comprehensive Google Colab notebook for using **Chatterbox TTS** by [Resemble AI](https://resemble.ai) - a state-of-the-art open-source text-to-speech model.

## 🌟 Features

The notebook includes:

- **🚀 Easy Setup**: One-click installation and model loading
- **🎯 Basic TTS**: Simple text-to-speech with default voice
- **🎭 Voice Cloning**: Upload audio samples to clone any voice
- **🎛️ Interactive Controls**: Real-time parameter adjustment with widgets
- **🔄 Voice Conversion**: Convert existing speech to different voices
- **💾 File Downloads**: Save generated audio to your device
- **🔍 Watermark Detection**: Verify responsible AI watermarking
- **📝 Example Texts**: Pre-written samples for different scenarios

## 🚀 Quick Start

1. **Open in Colab**: Click the "Open in Colab" badge above
2. **Enable GPU**: Go to Runtime → Change runtime type → Select GPU
3. **Run All Cells**: Runtime → Run all (or run cells individually)
4. **Upload Audio**: When prompted, upload audio files for voice cloning
5. **Experiment**: Try different texts and parameter combinations

## 🎛️ Parameters Explained

### Exaggeration (0.25 - 2.0)
- **0.5**: Neutral, natural speech
- **0.7-1.0**: More expressive and emotional
- **1.5+**: Highly dramatic (may be unstable)

### CFG Weight (0.0 - 1.0)
- **0.3**: Slower, more deliberate pacing
- **0.5**: Balanced (default)
- **0.8+**: Faster speech

### Temperature (0.1 - 2.0)
- **0.5**: Very consistent output
- **0.8**: Balanced randomness (default)
- **1.5+**: More creative but less predictable

## 💡 Tips for Best Results

### General Use:
- Default settings work well for most cases
- For fast speakers, lower CFG weight to ~0.3

### Expressive Speech:
- Use lower CFG weight (~0.3) + higher exaggeration (~0.7)
- Higher exaggeration speeds up speech; lower CFG compensates

### Voice Cloning:
- Use clear, 3-10 second audio samples
- Avoid background noise
- Single speaker recordings work best

## 📁 File Structure

```
chatterbox-colab/
├── chatterbox_colab.ipynb    # Main Colab notebook
├── COLAB_README.md          # This file
└── README.md                # Original project README
```

## 🔧 Requirements

The notebook automatically installs all dependencies:
- `chatterbox-tts`
- `ipywidgets`
- `resemble-perth`

## 🎯 Use Cases

- **Content Creation**: Voiceovers for videos, podcasts
- **Gaming**: Character voices and narration
- **Education**: Interactive learning materials
- **Accessibility**: Text-to-speech for visually impaired users
- **Prototyping**: Voice interface mockups
- **Entertainment**: Memes, creative projects

## 🔍 Watermarking

All generated audio includes Perth watermarks for responsible AI use. The notebook includes a section to detect these watermarks.

## 📚 Additional Resources

- [Original Chatterbox Repository](https://github.com/resemble-ai/chatterbox)
- [Demo Samples](https://resemble-ai.github.io/chatterbox_demopage/)
- [Hugging Face Space](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- [Discord Community](https://discord.gg/rJq9cRJBJ6)
- [Resemble AI](https://resemble.ai)

## ⚠️ Responsible Use

Please use this technology ethically and responsibly:
- Don't create deepfakes or impersonate others without consent
- Respect privacy and intellectual property rights
- Be transparent about AI-generated content
- Follow local laws and regulations

## 🤝 Contributing

Feel free to submit issues, suggestions, or improvements to make this notebook even better!

## 📄 License

This notebook is provided under the same MIT license as the original Chatterbox project.

---

**Made with ♥️ for the AI community**

*Happy voice synthesis! 🎤*
