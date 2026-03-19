"""
Database service for handling MongoDB connections and audio prompt retrieval
"""

import logging
import base64
import tempfile
import os
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import datetime

logger = logging.getLogger(__name__)

class AudioPromptDatabase:
    """Service for managing audio prompts from MongoDB"""
    
    def __init__(self, connection_string: str = "mongodb://admin:secret@mongodb.flashlit.ai:27017"):
        self.connection_string = connection_string
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        
    async def connect(self):
        """Initialize database connection"""
        try:
            logger.info("Connecting to MongoDB...")
            self.client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Get database and collection
            self.db = self.client.audio_sources_db
            self.collection = self.db.audio_prompts_a2flow
            
            # Log collection stats
            count = await self.collection.count_documents({})
            logger.info(f"Connected to audio_prompts_a2flow collection with {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def get_audio_prompt(self, actor_name: str, emotion: str) -> Optional[str]:
        """
        Fetch audio prompt by actor name and emotion
        
        Args:
            actor_name: Name of the actor
            emotion: Emotion type
            
        Returns:
            Path to temporary audio file, or None if not found
        """
        if self.collection is None:
            logger.error("Database not connected")
            return None
            
        try:
            logger.info(f"Searching for audio prompt: actor='{actor_name}', emotion='{emotion}'")
            
            # Query the database
            query = {
                "actor_name": {"$regex": f"^{actor_name}$", "$options": "i"},  # Case-insensitive exact match
                "emotion": {"$regex": f"^{emotion}$", "$options": "i"}         # Case-insensitive exact match
            }
            
            document = await self.collection.find_one(query)
            
            if not document:
                logger.warning(f"No audio prompt found for actor='{actor_name}', emotion='{emotion}'")
                return None
            
            logger.info(f"Found audio prompt: {document.get('original_file_name', 'unknown')}")
            
            # Extract base64 audio data
            audio_base64 = document.get('audio_base64')
            if not audio_base64:
                logger.error("No audio_base64 data in document")
                return None
            
            # Decode base64 and save to temporary file
            try:
                audio_data = base64.b64decode(audio_base64)
            except Exception as decode_error:
                logger.error(f"Failed to decode base64 audio: {decode_error}")
                return None
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(audio_data)
            temp_file.close()
            
            logger.info(f"Audio prompt saved to temporary file: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error fetching audio prompt: {e}")
            return None
    
    async def list_actors(self) -> List[str]:
        """Get list of available actors"""
        if self.collection is None:
            return []
            
        try:
            actors = await self.collection.distinct("actor_name")
            return sorted(actors) if actors else []
        except Exception as e:
            logger.error(f"Error fetching actors: {e}")
            return []
    
    async def list_emotions(self, actor_name: Optional[str] = None) -> List[str]:
        """Get list of available emotions, optionally filtered by actor"""
        if self.collection is None:
            return []
            
        try:
            query = {}
            if actor_name:
                query["actor_name"] = {"$regex": f"^{actor_name}$", "$options": "i"}
                
            emotions = await self.collection.distinct("emotion", query)
            return sorted(emotions) if emotions else []
        except Exception as e:
            logger.error(f"Error fetching emotions: {e}")
            return []
    
    async def get_audio_prompt_info(self, actor_name: str, emotion: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an audio prompt without downloading the audio
        
        Returns:
            Dictionary with prompt information or None if not found
        """
        if self.collection is None:
            return None
            
        try:
            query = {
                "actor_name": {"$regex": f"^{actor_name}$", "$options": "i"},
                "emotion": {"$regex": f"^{emotion}$", "$options": "i"}
            }
            
            document = await self.collection.find_one(
                query,
                {"audio_base64": 0}  # Exclude the large base64 field
            )
            
            if document:
                # Convert ObjectId to string for JSON serialization
                document["_id"] = str(document["_id"])
                
            return document
            
        except Exception as e:
            logger.error(f"Error fetching audio prompt info: {e}")
            return None

    async def add_audio_prompt(self, *, actor_name: str, emotion: str, transcription: str,
                               language_code: str, wav_bytes: bytes,
                               original_file_name: Optional[str] = None,
                               extra_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a new audio prompt to the database from raw WAV bytes.
        Stores base64-encoded audio and metadata.
        Returns inserted document id, or None on failure.
        """
        if self.collection is None:
            logger.error("Database not connected")
            return None

        try:
            # Encode to base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

            doc: Dict[str, Any] = {
                "actor_name": actor_name,
                "emotion": emotion,
                "transcription": transcription,
                "language_code": language_code,
                "audio_base64": audio_base64,
                "original_file_name": original_file_name or "uploaded.wav",
                "created_at": datetime.datetime.utcnow(),
            }
            if extra_metadata:
                doc.update({k: v for k, v in extra_metadata.items() if k not in doc})

            result = await self.collection.insert_one(doc)
            inserted_id = str(result.inserted_id)
            logger.info(f"Inserted audio prompt for '{actor_name}'/'{emotion}' with id {inserted_id}")
            return inserted_id
        except Exception as e:
            logger.error(f"Failed to insert audio prompt: {e}")
            return None

# Global database instance
audio_db = AudioPromptDatabase()

async def cleanup_temp_file(file_path: str):
    """Clean up temporary audio file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

async def get_audio_prompt_path(actor_name: Optional[str], emotion: Optional[str], 
                               audio_prompt_path: Optional[str] = None) -> Optional[str]:
    """
    Get audio prompt path - either from database or from provided path
    
    Args:
        actor_name: Actor name for database lookup
        emotion: Emotion for database lookup  
        audio_prompt_path: Direct file path (takes precedence)
        
    Returns:
        Path to audio file or None
    """
    # If direct path is provided, use it
    if audio_prompt_path:
        return audio_prompt_path
    
    # If actor and emotion are provided, fetch from database
    if actor_name and emotion:
        return await audio_db.get_audio_prompt(actor_name, emotion)
    
    # No audio prompt available
    return None
