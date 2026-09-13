import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from chatterbox.tts import ChatterboxTTS, Conditionals


@dataclass
class VoiceProfile:
    """
    Metadata for a saved voice profile.
    """
    name: str
    audio_sources: List[str]
    lang: str = 'en'
    gender: Optional[str] = None
    description: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


class VoiceBank:
    """
    Manages saved voice profiles and their corresponding Conditionals.
    """

    def __init__(self, storage_dir: str = './voices'):
        """
        Initialize the VoiceBank.

        Args:
            storage_dir (str): Directory where voice profiles will be stored.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def voice_exists(self, name: str) -> bool:
        """
        Check if a voice profile exists.

        Args:
            name (str): Name of the voice.

        Returns:
            bool: True if the voice exists, False otherwise.
        """
        pt_path = os.path.join(self.storage_dir, f"{name}.pt")
        json_path = os.path.join(self.storage_dir, f"{name}.json")
        return os.path.exists(pt_path) and os.path.exists(json_path)

    def register_voice(
        self,
        name: str,
        audio_paths: List[str],
        model: ChatterboxTTS,
        lang: str = 'en',
        gender: Optional[str] = None,
        description: Optional[str] = None
    ) -> VoiceProfile:
        """
        Register a new voice profile from audio files.

        Args:
            name (str): Name of the voice.
            audio_paths (List[str]): List of audio file paths.
            model (ChatterboxTTS): The TTS model to use for extracting conditionals.
            lang (str): Language of the voice.
            gender (str, optional): Gender of the voice ('male'/'female'/'other').
            description (str, optional): Description of the voice.

        Returns:
            VoiceProfile: The created voice profile.

        Raises:
            ValueError: If the voice name already exists or audio paths are empty.
            FileNotFoundError: If any audio path does not exist.
        """
        if self.voice_exists(name):
            raise ValueError(f"Voice profile '{name}' already exists.")

        if not audio_paths:
            raise ValueError("At least one audio path must be provided.")

        for path in audio_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

        # Extract conditionals
        conds = model.prepare_conditionals(audio_paths)

        # Save conditionals
        pt_path = os.path.join(self.storage_dir, f"{name}.pt")
        conds.save(pt_path)

        # Create and save profile
        profile = VoiceProfile(
            name=name,
            audio_sources=audio_paths,
            lang=lang,
            gender=gender,
            description=description
        )

        json_path = os.path.join(self.storage_dir, f"{name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=4, ensure_ascii=False)

        return profile

    def list_voices(self) -> List[VoiceProfile]:
        """
        List all registered voice profiles.

        Returns:
            List[VoiceProfile]: A list of all available voice profiles.
        """
        profiles = []
        if not os.path.exists(self.storage_dir):
            return profiles

        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                name = filename[:-5]
                if self.voice_exists(name):
                    json_path = os.path.join(self.storage_dir, filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            profiles.append(VoiceProfile.from_dict(data))
                    except Exception as e:
                        print(f"Failed to load profile {filename}: {e}")
        return profiles

    def get_voice(self, name: str) -> Tuple[VoiceProfile, Conditionals]:
        """
        Get a voice profile and its Conditionals by name.

        Args:
            name (str): Name of the voice.

        Returns:
            Tuple[VoiceProfile, Conditionals]: The profile and its corresponding conditionals.

        Raises:
            FileNotFoundError: If the voice profile does not exist.
        """
        if not self.voice_exists(name):
            raise FileNotFoundError(f"Voice profile '{name}' does not exist.")

        json_path = os.path.join(self.storage_dir, f"{name}.json")
        pt_path = os.path.join(self.storage_dir, f"{name}.pt")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            profile = VoiceProfile.from_dict(data)

        conds = Conditionals.load(pt_path)

        return profile, conds

    def delete_voice(self, name: str) -> bool:
        """
        Delete a voice profile by name.

        Args:
            name (str): Name of the voice.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        if not self.voice_exists(name):
            return False

        json_path = os.path.join(self.storage_dir, f"{name}.json")
        pt_path = os.path.join(self.storage_dir, f"{name}.pt")

        try:
            os.remove(json_path)
            os.remove(pt_path)
            return True
        except OSError:
            return False
