"""
Enhanced Chatterbox TTS with Audiobook Features

This app combines the original Chatterbox TTS with advanced audiobook creation features
including voice library management, long text processing, and multi-voice support.
"""

import gradio as gr
import torch
import numpy as np
import json
import os
import shutil
import tempfile
import time
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our enhanced audiobook processing
from src.chatterbox.audiobook_processing import AudiobookProcessor

# Try importing TTS modules
try:
    from src.chatterbox.tts import ChatterboxTTS
    from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS modules not available - {e}")
    TTS_AVAILABLE = False

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice to avoid CUDA issues

# Initialize audiobook processor
audiobook_processor = AudiobookProcessor()

class ChatterboxAudiobookApp:
    def __init__(self):
        self.tts_model = None
        self.multilingual_model = None
        self.current_voice_profile = None
        
    def initialize_models(self):
        """Initialize TTS models if available."""
        if TTS_AVAILABLE and self.tts_model is None:
            try:
                self.tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
                self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
                return True
            except Exception as e:
                print(f"Error initializing models: {e}")
                return False
        return TTS_AVAILABLE
    
    def generate_speech(self, text: str, voice_profile: str = None, 
                       language: str = "en", **kwargs) -> tuple:
        """Generate speech with enhanced audiobook features."""
        if not self.initialize_models():
            return None, "❌ TTS models not available"
        
        if not text.strip():
            return None, "❌ Please provide text to synthesize"
        
        # Load voice profile if provided
        voice_config = {}
        if voice_profile:
            voice_config = audiobook_processor.load_voice_profile(voice_profile)
        
        # Merge voice settings with provided parameters
        params = {
            'exaggeration': voice_config.get('exaggeration', 0.5),
            'cfg_weight': voice_config.get('cfg_weight', 0.5),
            'temperature': voice_config.get('temperature', 0.8),
            **kwargs
        }
        
        try:
            if language == "en":
                audio = self.tts_model.generate(text, **params)
            else:
                audio = self.multilingual_model.generate(text, language_id=language, **params)
            
            # Convert to numpy array for Gradio
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            return (self.tts_model.sr, audio), "✅ Speech generated successfully!"
            
        except Exception as e:
            return None, f"❌ Error generating speech: {str(e)}"
    
    def create_audiobook(self, text_content: str, project_name: str, 
                        selected_voice: str, language: str = "en",
                        enable_chunking: bool = True, **kwargs) -> tuple:
        """Create an audiobook from text with chunking support."""
        
        # Validate input
        is_valid, error_msg = audiobook_processor.validate_audiobook_input(
            text_content, project_name
        )
        if not is_valid:
            return None, error_msg
        
        if not self.initialize_models():
            return None, "❌ TTS models not available"
        
        # Load voice profile
        voice_config = audiobook_processor.load_voice_profile(selected_voice)
        
        # Prepare parameters
        params = {
            'exaggeration': voice_config.get('exaggeration', 0.5),
            'cfg_weight': voice_config.get('cfg_weight', 0.5),
            'temperature': voice_config.get('temperature', 0.8),
            **kwargs
        }
        
        # Process text
        if enable_chunking:
            chunks = audiobook_processor.adaptive_chunk_text(text_content)
        else:
            chunks = [text_content]
        
        # Generate audio chunks
        audio_chunks = []
        sample_rate = self.tts_model.sr
        
        for i, chunk in enumerate(chunks):
            try:
                if language == "en":
                    audio = self.tts_model.generate(chunk, **params)
                else:
                    audio = self.multilingual_model.generate(chunk, language_id=language, **params)
                
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                audio_chunks.append(audio)
                
            except Exception as e:
                return None, f"❌ Error generating chunk {i+1}: {str(e)}"
        
        # Combine audio chunks
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            return (sample_rate, combined_audio), f"✅ Audiobook '{project_name}' created successfully!"
        
        return None, "❌ No audio generated"
    
    def create_multi_voice_audiobook(self, text_content: str, project_name: str,
                                   voice_mappings: dict, language: str = "en") -> tuple:
        """Create multi-voice audiobook with character voice mapping."""
        
        if not self.initialize_models():
            return None, "❌ TTS models not available"
        
        # Parse multi-voice text
        segments = audiobook_processor.add_pause_segments(text_content)
        
        audio_chunks = []
        sample_rate = self.tts_model.sr
        
        for i, segment in enumerate(segments):
            character = segment['character']
            text = segment['text']
            
            # Get voice profile for character
            voice_name = voice_mappings.get(character, list(voice_mappings.values())[0])
            voice_config = audiobook_processor.load_voice_profile(voice_name)
            
            params = {
                'exaggeration': voice_config.get('exaggeration', 0.5),
                'cfg_weight': voice_config.get('cfg_weight', 0.5),
                'temperature': voice_config.get('temperature', 0.8)
            }
            
            try:
                # Generate speech for this segment
                if language == "en":
                    audio = self.tts_model.generate(text, **params)
                else:
                    audio = self.multilingual_model.generate(text, language_id=language, **params)
                
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                audio_chunks.append(audio)
                
                # Add pause if needed
                if segment['pause_duration'] > 0:
                    pause_samples = int(segment['pause_duration'] * sample_rate)
                    pause_audio = np.zeros(pause_samples)
                    audio_chunks.append(pause_audio)
                
            except Exception as e:
                return None, f"❌ Error generating segment {i+1} for {character}: {str(e)}"
        
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            return (sample_rate, combined_audio), f"✅ Multi-voice audiobook '{project_name}' created!"
        
        return None, "❌ No audio generated"

# Initialize app
app = ChatterboxAudiobookApp()

# Define Gradio interface
def create_interface():
    with gr.Blocks(title="Chatterbox Audiobook Studio", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎧 Chatterbox Audiobook Studio
        
        **Professional audiobook creation with advanced TTS, voice cloning, and multi-character support.**
        """)
        
        with gr.Tabs():
            # Basic TTS Tab
            with gr.TabItem("🎤 Basic TTS"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="Enter your text here...",
                            lines=4,
                            max_lines=10
                        )
                        
                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                choices=[
                                    ("English", "en"), ("Spanish", "es"), ("French", "fr"),
                                    ("German", "de"), ("Chinese", "zh"), ("Japanese", "ja"),
                                    ("Korean", "ko"), ("Italian", "it"), ("Portuguese", "pt"),
                                    ("Russian", "ru"), ("Arabic", "ar"), ("Hindi", "hi")
                                ],
                                value="en",
                                label="Language"
                            )
                            
                            voice_dropdown = gr.Dropdown(
                                choices=audiobook_processor.list_voice_profiles(),
                                label="Voice Profile (Optional)",
                                value=None
                            )
                    
                    with gr.Column():
                        exaggeration_slider = gr.Slider(0.0, 2.0, 0.5, label="Exaggeration")
                        cfg_slider = gr.Slider(0.1, 1.0, 0.5, label="CFG Weight")
                        temp_slider = gr.Slider(0.1, 2.0, 0.8, label="Temperature")
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
                audio_output = gr.Audio(label="Generated Audio")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            # Audiobook Creation Tab
            with gr.TabItem("📚 Audiobook Creation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audiobook_text = gr.Textbox(
                            label="Text Content",
                            placeholder="Paste your book content here...",
                            lines=8,
                            max_lines=20
                        )
                        
                        text_file = gr.File(
                            label="Or upload text file",
                            file_types=[".txt", ".md"]
                        )
                        
                        project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="My Audiobook"
                        )
                    
                    with gr.Column():
                        audiobook_voice = gr.Dropdown(
                            choices=audiobook_processor.list_voice_profiles(),
                            label="Voice Profile",
                            value=None
                        )
                        
                        audiobook_language = gr.Dropdown(
                            choices=[
                                ("English", "en"), ("Spanish", "es"), ("French", "fr"),
                                ("German", "de"), ("Chinese", "zh"), ("Japanese", "ja"),
                                ("Korean", "ko"), ("Italian", "it"), ("Portuguese", "pt"),
                                ("Russian", "ru"), ("Arabic", "ar"), ("Hindi", "hi"),
                                ("Bangla", "bn"), ("Danish", "da"), ("Greek", "el"),
                                ("Hebrew", "he"), ("Finnish", "fi"), ("Malay", "ms"),
                                ("Dutch", "nl"), ("Norwegian", "no"), ("Polish", "pl"),
                                ("Swedish", "sv"), ("Swahili", "sw"), ("Turkish", "tr")
                            ],
                            value="en",
                            label="Language"
                        )
                        
                        enable_chunking = gr.Checkbox(
                            label="Enable Text Chunking",
                            value=True
                        )
                
                create_audiobook_btn = gr.Button("📖 Create Audiobook", variant="primary")
                audiobook_audio = gr.Audio(label="Generated Audiobook")
                audiobook_status = gr.Textbox(label="Status", interactive=False)
            
            # Voice Library Tab
            with gr.TabItem("📚 Voice Library"):
                gr.Markdown("""
                ### 🎭 Voice Library Management
                
                Create and manage voice profiles for consistent character voices across your audiobooks.
                """)
                
                with gr.Row():
                    with gr.Column():
                        voice_name = gr.Textbox(label="Voice Name")
                        display_name = gr.Textbox(label="Display Name")
                        description = gr.Textbox(label="Description", lines=2)
                        reference_audio = gr.Audio(label="Reference Audio (10-30 seconds)")
                        
                        with gr.Row():
                            ex_voice = gr.Slider(0.0, 2.0, 0.5, label="Exaggeration")
                            cfg_voice = gr.Slider(0.1, 1.0, 0.5, label="CFG Weight")
                            temp_voice = gr.Slider(0.1, 2.0, 0.8, label="Temperature")
                    
                    with gr.Column():
                        existing_voices = gr.Dropdown(
                            choices=audiobook_processor.list_voice_profiles(),
                            label="Existing Voices"
                        )
                        
                        voice_info = gr.Textbox(label="Voice Details", lines=4, interactive=False)
                        
                        with gr.Row():
                            save_voice_btn = gr.Button("💾 Save Voice", variant="primary")
                            delete_voice_btn = gr.Button("🗑️ Delete Voice")
                            refresh_voices_btn = gr.Button("🔄 Refresh")
            
            # Multi-Voice Tab
            with gr.TabItem("🎭 Multi-Voice Creation"):
                gr.Markdown("""
                ### 🎭 Multi-Voice Audiobook Creation
                
                Create audiobooks with multiple character voices using the format:
                ```
                [Character1] This is dialogue for character 1.
                
                [Character2] This is dialogue for character 2.
                
                [Narrator] This is narration text.
                ```
                
                **Tip:** Use line breaks to add natural pauses between dialogue.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        multi_text = gr.Textbox(
                            label="Multi-Voice Text",
                            placeholder="[Character1] Hello there!\n\n[Character2] How are you?",
                            lines=10,
                            max_lines=25
                        )
                        
                        multi_project = gr.Textbox(
                            label="Project Name",
                            placeholder="My Multi-Voice Audiobook"
                        )
                    
                    with gr.Column():
                        available_voices = audiobook_processor.list_voice_profiles()
                        
                        voice_mapping = gr.Dataframe(
                            headers=["Character", "Voice Profile"],
                            datatype=["str", "str"],
                            row_count=5,
                            col_count=(2, "fixed"),
                            label="Character Voice Mapping"
                        )
                        
                        multi_language = gr.Dropdown(
                            choices=[
                                ("English", "en"), ("Spanish", "es"), ("French", "fr"),
                                ("German", "de"), ("Chinese", "zh"), ("Japanese", "ja"),
                                ("Bangla", "bn"), ("Arabic", "ar"), ("Hindi", "hi"),
                                ("Portuguese", "pt"), ("Russian", "ru"), ("Korean", "ko")
                            ],
                            value="en",
                            label="Language"
                        )
                
                create_multi_btn = gr.Button("🎭 Create Multi-Voice Audiobook", variant="primary")
                multi_audio = gr.Audio(label="Generated Multi-Voice Audiobook")
                multi_status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        def update_voice_list():
            voices = audiobook_processor.list_voice_profiles()
            return gr.Dropdown(choices=voices), gr.Dropdown(choices=voices), gr.Dropdown(choices=voices)
        
        def load_voice_details(voice_name):
            if not voice_name:
                return ""
            
            config = audiobook_processor.load_voice_profile(voice_name)
            if not config:
                return "❌ Voice not found"
            
            return f"""
Name: {config.get('display_name', voice_name)}
Description: {config.get('description', 'No description')}
Created: {time.strftime('%Y-%m-%d', time.localtime(float(config.get('created_date', 0))))}
Settings: Exaggeration={config.get('exaggeration', 0.5)}, CFG={config.get('cfg_weight', 0.5)}
            """
        
        # Basic TTS
        generate_btn.click(
            fn=lambda text, lang, voice, ex, cfg, temp: app.generate_speech(
                text, voice, lang, exaggeration=ex, cfg_weight=cfg, temperature=temp
            ),
            inputs=[text_input, language_dropdown, voice_dropdown, 
                   exaggeration_slider, cfg_slider, temp_slider],
            outputs=[audio_output, status_text]
        )
        
        # Audiobook creation
        def process_audiobook(text, file_content, project, voice, lang, chunking):
            if file_content is not None:
                content, status = audiobook_processor.load_text_file(file_content.name)
                if not content:
                    return None, status
                text = content
            
            return app.create_audiobook(text, project, voice, lang, chunking)
        
        create_audiobook_btn.click(
            fn=process_audiobook,
            inputs=[audiobook_text, text_file, project_name, 
                   audiobook_voice, audiobook_language, enable_chunking],
            outputs=[audiobook_audio, audiobook_status]
        )
        
        # Voice library management
        def save_voice(voice, display, desc, ref_audio, ex, cfg, temp):
            if not voice.strip():
                return "❌ Please provide a voice name"
            
            audio_path = None
            if ref_audio is not None:
                audio_path = ref_audio
            
            return audiobook_processor.save_voice_profile(
                voice.strip(), display, desc, audio_path, ex, cfg, temp
            )
        
        save_voice_btn.click(
            fn=save_voice,
            inputs=[voice_name, display_name, description, reference_audio,
                   ex_voice, cfg_voice, temp_voice],
            outputs=[voice_info]
        )
        
        delete_voice_btn.click(
            fn=lambda voice: audiobook_processor.delete_voice_profile(voice),
            inputs=[existing_voices],
            outputs=[voice_info]
        ).then(update_voice_list, outputs=[voice_dropdown, audiobook_voice, existing_voices])
        
        refresh_voices_btn.click(
            fn=update_voice_list,
            outputs=[voice_dropdown, audiobook_voice, existing_voices]
        )
        
        existing_voices.change(
            fn=load_voice_details,
            inputs=[existing_voices],
            outputs=[voice_info]
        )
        
        # Multi-voice creation
        def create_multi_voice(text, project, mapping_df, lang):
            voice_dict = {}
            for row in mapping_df:
                if row[0] and row[1]:
                    voice_dict[row[0]] = row[1]
            
            if not voice_dict:
                return None, "❌ Please map characters to voices"
            
            return app.create_multi_voice_audiobook(text, project, voice_dict, lang)
        
        create_multi_btn.click(
            fn=create_multi_voice,
            inputs=[multi_text, multi_project, voice_mapping, multi_language],
            outputs=[multi_audio, multi_status]
        )
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=False,
        quiet=False
    )
