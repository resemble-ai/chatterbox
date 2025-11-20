import sys
import re
import torch
import torchaudio as ta
from pathlib import Path
from safetensors.torch import load_file
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLineEdit, QTextEdit, 
                               QComboBox, QLabel, QGroupBox, QFileDialog, 
                               QMessageBox, QCheckBox, QDoubleSpinBox)
from PySide6.QtCore import QThread, Signal, QSettings
from chatterbox_git.src.chatterbox import mtl_tts


def split_into_sentences(text):
    """Simple sentence splitter for batching"""
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'(?<!\d)[.!?](?=\s*[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class TTSThread(QThread):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, text, language_id, device, model_path, output_path, 
                 audio_prompt_path=None, use_batching=False, output_format="wav",
                 temperature=0.8, cfg_weight=0.3):
        super().__init__()
        self.text = text
        self.language_id = language_id
        self.device = device
        self.model_path = model_path
        self.output_path = output_path
        self.audio_prompt_path = audio_prompt_path
        self.use_batching = use_batching
        self.output_format = output_format.lower()
        self.temperature = temperature
        self.cfg_weight = cfg_weight
        
    def run(self):
        try:
            # Load model (EXACTLY like reference)
            self.progress.emit("Loading model...")
            multilingual_model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            
            self.progress.emit("Loading language weights...")
            t3_state = load_file(self.model_path, device=self.device)
            multilingual_model.t3.load_state_dict(t3_state)
            multilingual_model.t3.to(self.device).eval()
            
            # Ensure output directory exists
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate audio
            if self.use_batching:
                self.progress.emit("Splitting into sentences...")
                sentences = split_into_sentences(self.text)
                
                if len(sentences) == 0:
                    raise ValueError("No sentences found in text")
                
                self.progress.emit(f"Processing {len(sentences)} sentences...")
                wav_chunks = []
                
                for i, sentence in enumerate(sentences, 1):
                    self.progress.emit(f"Generating sentence {i}/{len(sentences)}: {sentence[:50]}...")
                    
                    wav_chunk = multilingual_model.generate(
                        sentence, 
                        language_id=self.language_id,
                        audio_prompt_path=self.audio_prompt_path,
                        temperature=self.temperature,
                        cfg_weight=self.cfg_weight
                    )
                    wav_chunks.append(wav_chunk)
                    
                    # Progressive saving - save accumulated chunks after each sentence
                    self.progress.emit(f"Saving progress ({i}/{len(sentences)})...")
                    wav_so_far = torch.cat(wav_chunks, dim=-1)
                    if self.output_format == "mp3":
                        ta.save(self.output_path, wav_so_far, multilingual_model.sr, format="mp3")
                    else:
                        ta.save(self.output_path, wav_so_far, multilingual_model.sr)
                
                # Final combined audio (already saved in loop)
                wav = wav_so_far
            else:
                self.progress.emit("Generating audio...")
                wav = multilingual_model.generate(
                    self.text, 
                    language_id=self.language_id,
                    audio_prompt_path=self.audio_prompt_path,
                    temperature=self.temperature,
                    cfg_weight=self.cfg_weight
                )
                
                # Save audio
                self.progress.emit("Saving audio...")
                if self.output_format == "mp3":
                    ta.save(self.output_path, wav, multilingual_model.sr, format="mp3")
                else:
                    ta.save(self.output_path, wav, multilingual_model.sr)
            
            self.finished.emit(self.output_path)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ChatterboxTTSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS - Simple")
        self.setMinimumWidth(600)
        self.tts_thread = None
        self.settings = QSettings("ChatterboxTTS", "SimpleGUI")
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Model Settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Language
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language ID:"))
        self.language_edit = QLineEdit("cs")
        self.language_edit.setPlaceholderText("e.g., cs, en, de")
        lang_layout.addWidget(self.language_edit)
        model_layout.addLayout(lang_layout)
        
        # Device
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "mps"])
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        model_layout.addLayout(device_layout)
        
        # Model Path
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model (.safetensors):"))
        self.model_path_edit = QLineEdit("t3_cs.safetensors")
        model_path_layout.addWidget(self.model_path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(browse_btn)
        model_layout.addLayout(model_path_layout)
        
        # Reference Voice (Audio Prompt)
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Reference Voice:"))
        self.prompt_path_edit = QLineEdit()
        self.prompt_path_edit.setPlaceholderText("Optional - leave empty for default voice")
        prompt_layout.addWidget(self.prompt_path_edit)
        browse_prompt_btn = QPushButton("Browse...")
        browse_prompt_btn.clicked.connect(self.browse_prompt)
        prompt_layout.addWidget(browse_prompt_btn)
        model_layout.addLayout(prompt_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Text Input
        text_group = QGroupBox("Text to Synthesize")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text here...")
        self.text_edit.setMaximumHeight(120)
        text_layout.addWidget(self.text_edit)
        
        # Batching option
        self.batch_checkbox = QCheckBox("Use sentence batching (for longer texts with short sentences)")
        text_layout.addWidget(self.batch_checkbox)
        
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Generation Parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.05, 5.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(0.8)
        temp_layout.addWidget(self.temperature_spin)
        temp_layout.addWidget(QLabel("(Lower = more consistent, Higher = more varied)"))
        temp_layout.addStretch()
        params_layout.addLayout(temp_layout)
        
        # CFG Weight / Pace
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("CFG Weight / Pace:"))
        self.cfg_weight_spin = QDoubleSpinBox()
        self.cfg_weight_spin.setRange(0.0, 1.0)
        self.cfg_weight_spin.setSingleStep(0.05)
        self.cfg_weight_spin.setValue(0.3)
        cfg_layout.addWidget(self.cfg_weight_spin)
        cfg_layout.addWidget(QLabel("(Controls generation guidance and pacing)"))
        cfg_layout.addStretch()
        params_layout.addLayout(cfg_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Output Settings
        output_group = QGroupBox("Output Settings")
        output_group_layout = QVBoxLayout()
        
        # Format selector - MORE PROMINENT
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["WAV", "MP3"])
        self.format_combo.setMinimumWidth(100)
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        output_group_layout.addLayout(format_layout)
        
        # Output Path
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output File:"))
        self.output_path_edit = QLineEdit("output.wav")
        output_layout.addWidget(self.output_path_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(browse_output_btn)
        output_group_layout.addLayout(output_layout)
        
        output_group.setLayout(output_group_layout)
        layout.addWidget(output_group)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Generate Button
        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate_speech)
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
    
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "SafeTensors (*.safetensors);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_prompt(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Voice", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if file_path:
            self.prompt_path_edit.setText(file_path)
    
    def browse_output(self):
        format_filter = "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "", format_filter
        )
        if file_path:
            self.output_path_edit.setText(file_path)
            # Auto-detect format from extension
            if file_path.lower().endswith('.mp3'):
                self.format_combo.setCurrentText("MP3")
            else:
                self.format_combo.setCurrentText("WAV")
    
    def on_format_changed(self, format_text):
        """Update file extension when format changes"""
        current_path = Path(self.output_path_edit.text())
        if format_text == "MP3":
            new_path = current_path.with_suffix('.mp3')
        else:
            new_path = current_path.with_suffix('.wav')
        self.output_path_edit.setText(str(new_path))
    
    def load_settings(self):
        """Load all settings from QSettings"""
        self.language_edit.setText(self.settings.value("language", "cs"))
        self.device_combo.setCurrentText(self.settings.value("device", "cuda"))
        self.model_path_edit.setText(self.settings.value("model_path", "t3_cs.safetensors"))
        self.prompt_path_edit.setText(self.settings.value("prompt_path", ""))
        self.temperature_spin.setValue(float(self.settings.value("temperature", 0.8)))
        self.cfg_weight_spin.setValue(float(self.settings.value("cfg_weight", 0.3)))
        self.batch_checkbox.setChecked(self.settings.value("use_batching", False, type=bool))
        self.output_path_edit.setText(self.settings.value("output_path", "output.wav"))
        self.format_combo.setCurrentText(self.settings.value("format", "WAV"))
        self.text_edit.setText(self.settings.value("text", ""))
    
    def save_settings(self):
        """Save all settings to QSettings"""
        self.settings.setValue("language", self.language_edit.text())
        self.settings.setValue("device", self.device_combo.currentText())
        self.settings.setValue("model_path", self.model_path_edit.text())
        self.settings.setValue("prompt_path", self.prompt_path_edit.text())
        self.settings.setValue("temperature", self.temperature_spin.value())
        self.settings.setValue("cfg_weight", self.cfg_weight_spin.value())
        self.settings.setValue("use_batching", self.batch_checkbox.isChecked())
        self.settings.setValue("output_path", self.output_path_edit.text())
        self.settings.setValue("format", self.format_combo.currentText())
        self.settings.setValue("text", self.text_edit.toPlainText())
    
    def generate_speech(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text")
            return
        
        model_path = self.model_path_edit.text()
        if not Path(model_path).exists():
            QMessageBox.warning(self, "Error", f"Model file not found: {model_path}")
            return
        
        audio_prompt_path = self.prompt_path_edit.text().strip() or None
        if audio_prompt_path and not Path(audio_prompt_path).exists():
            QMessageBox.warning(self, "Error", f"Reference voice file not found: {audio_prompt_path}")
            return
        
        # Save settings before generation
        self.save_settings()
        
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Starting generation...")
        
        self.tts_thread = TTSThread(
            text=text,
            language_id=self.language_edit.text(),
            device=self.device_combo.currentText(),
            model_path=model_path,
            output_path=self.output_path_edit.text(),
            audio_prompt_path=audio_prompt_path,
            use_batching=self.batch_checkbox.isChecked(),
            output_format=self.format_combo.currentText(),
            temperature=self.temperature_spin.value(),
            cfg_weight=self.cfg_weight_spin.value()
        )
        
        self.tts_thread.finished.connect(self.on_finished)
        self.tts_thread.error.connect(self.on_error)
        self.tts_thread.progress.connect(self.on_progress)
        self.tts_thread.start()
    
    def on_progress(self, message):
        self.status_label.setText(message)
    
    def on_finished(self, output_path):
        self.generate_btn.setEnabled(True)
        self.status_label.setText(f"Complete! Saved to: {output_path}")
        QMessageBox.information(self, "Success", f"Audio saved to:\n{output_path}")
    
    def on_error(self, error_message):
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Generation failed")
        QMessageBox.critical(self, "Error", f"Error:\n{error_message}")
    
    def closeEvent(self, event):
        """Save settings when window is closed"""
        self.save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ChatterboxTTSGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
