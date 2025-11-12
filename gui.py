import sys
import torch
import torchaudio as ta
from pathlib import Path
from safetensors.torch import load_file
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLineEdit, QTextEdit, 
                               QComboBox, QLabel, QGroupBox, QFileDialog, 
                               QMessageBox, QProgressBar)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtCore import QUrl
from chatterbox_git.src.chatterbox import mtl_tts

class TTSThread(QThread):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, text, language_id, device, model_path, output_path, audio_prompt_path=None):
        super().__init__()
        self.text = text
        self.language_id = language_id
        self.device = device
        self.model_path = model_path
        self.output_path = output_path
        self.audio_prompt_path = audio_prompt_path
        
    def run(self):
        try:
            self.progress.emit("Loading multilingual model...")
            model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            
            self.progress.emit(f"Loading language weights...")
            t3_state = load_file(self.model_path, device=self.device)
            model.t3.load_state_dict(t3_state)
            model.t3.to(self.device).eval()
            
            self.progress.emit("Generating speech...")
            # CORRECT: Pass audio_prompt_path directly to generate()
            wav = model.generate(
                self.text, 
                language_id=self.language_id,
                audio_prompt_path=self.audio_prompt_path  # Direct path reference
            )
            
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            ta.save(self.output_path, wav, model.sr)
            
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))

class ChatterboxTTSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox Multilingual TTS")
        self.setMinimumSize(700, 450)
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.init_ui()
        
    # === CORRECT HELPER METHODS ===
    
    def browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "SafeTensors (*.safetensors)")
        if path:
            self.model_path_edit.setText(path)
            
    def browse_prompt_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            self.prompt_path_edit.setText(path)
            
    def browse_output_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV (*.wav)")
        if path:
            if not path.endswith('.wav'):
                path += '.wav'
            self.output_path_edit.setText(path)
        
    def generate_speech(self):
        # Validate inputs
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter text to synthesize.")
            return
            
        model_path = self.model_path_edit.text().strip()
        if not Path(model_path).exists():
            QMessageBox.warning(self, "Error", f"Model file not found:\n{model_path}")
            return
            
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Error", "Please specify an output path")
            return
        
        # Get audio prompt path (if any)
        audio_prompt_path = self.prompt_path_edit.text().strip()
        if audio_prompt_path and not Path(audio_prompt_path).exists():
            QMessageBox.warning(self, "Error", f"Reference audio not found:\n{audio_prompt_path}")
            return
        
        # Disable UI and start generation
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Generating...")
        
        self.tts_thread = TTSThread(
            text, 
            self.lang_combo.currentText(), 
            self.device_combo.currentText(), 
            model_path, 
            output_path, 
            audio_prompt_path if audio_prompt_path else None
        )
        self.tts_thread.finished.connect(self.on_finished)
        self.tts_thread.error.connect(self.on_error)
        self.tts_thread.progress.connect(self.status_label.setText)
        self.tts_thread.start()
        
    def on_finished(self, path):
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Saved: {Path(path).name}")
        self.play_btn.setEnabled(True)
        
    def on_error(self, error):
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Generation Error", error)
        
    def set_ui_enabled(self, enabled):
        widgets = [self.generate_btn, self.device_combo, self.lang_combo, self.text_edit, 
                  self.model_path_edit, self.browse_model_btn, self.prompt_path_edit, 
                  self.browse_prompt_btn, self.output_path_edit, self.browse_output_btn]
        for widget in widgets:
            widget.setEnabled(enabled)
            
    def play_audio(self):
        path = self.output_path_edit.text()
        if Path(path).exists():
            self.media_player.setSource(QUrl.fromLocalFile(path))
            self.media_player.play()
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Error", "Audio file not found. Generate it first.")
            
    def stop_audio(self):
        self.media_player.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    # === UI SETUP ===
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Device Settings
        device_group = QGroupBox("Device Settings")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + 
                                   (["mps"] if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else []))
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Model Settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["cs", "en", "de", "es", "fr", "it", "pl", "pt", "ru", "tr", "ja", "ko", "zh"])
        self.lang_combo.setCurrentText("cs")
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addStretch()
        model_layout.addLayout(lang_layout)
        
        # Language model path
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to language model (e.g., t3_cs.safetensors)...")
        model_path_layout.addWidget(self.model_path_edit)
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        model_path_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(model_path_layout)
        
        # Audio prompt path (Speaker conditioning)
        prompt_group = QGroupBox("Speaker Reference (Optional)")
        prompt_layout = QVBoxLayout()
        
        prompt_path_layout = QHBoxLayout()
        self.prompt_path_edit = QLineEdit()
        self.prompt_path_edit.setPlaceholderText("Path to reference audio file (optional)...")
        prompt_path_layout.addWidget(self.prompt_path_edit)
        self.browse_prompt_btn = QPushButton("Browse...")
        self.browse_prompt_btn.clicked.connect(self.browse_prompt_file)
        prompt_path_layout.addWidget(self.browse_prompt_btn)
        prompt_layout.addLayout(prompt_path_layout)
        
        prompt_group.setLayout(prompt_layout)
        model_layout.addWidget(prompt_group)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Text Input
        text_group = QGroupBox("Text to Speech")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text here...")
        self.text_edit.setText("Dobrý den, vítáme vás v našem testu syntézy řeči")
        text_layout.addWidget(self.text_edit)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(str(Path.cwd() / "output.wav"))
        output_path_layout.addWidget(self.output_path_edit)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_file)
        output_path_layout.addWidget(self.browse_output_btn)
        output_layout.addLayout(output_path_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate_speech)
        button_layout.addWidget(self.generate_btn)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        button_layout.addWidget(self.play_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

def main():
    app = QApplication(sys.argv)
    window = ChatterboxTTSGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
