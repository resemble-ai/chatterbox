"""
Async batch processing queue for TTS requests.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import io
import torchaudio

@dataclass
class TTSJob:
    id: uuid.UUID
    text: str
    lang: str
    voice_ref: Optional[str]
    model_type: str
    emotion: Optional[str]
    status: str = 'pending'
    result: Optional[bytes] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class BatchProcessor:
    def __init__(self, models: Dict[str, Any], max_queue_size: int = 100):
        self.models = models
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: Dict[str, TTSJob] = {}
        self.worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def submit(self, text: str, lang: str = 'en', model_type: str = 'multilingual', 
                     voice_ref: Optional[str] = None, emotion: Optional[str] = None) -> str:
        job_id = uuid.uuid4()
        job = TTSJob(
            id=job_id,
            text=text,
            lang=lang,
            voice_ref=voice_ref,
            model_type=model_type,
            emotion=emotion
        )
        self.jobs[str(job_id)] = job
        await self.queue.put(str(job_id))
        return str(job_id)

    async def get_status(self, job_id: str) -> TTSJob:
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self.jobs[job_id]

    async def get_result(self, job_id: str) -> bytes:
        job = await self.get_status(job_id)
        if job.status != 'done':
            raise ValueError(f"Job is not done, current status: {job.status}")
        if job.result is None:
            raise ValueError("Job done but no result available.")
        return job.result

    async def start_worker(self):
        self._stop_event.clear()
        self.worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._stop_event.set()
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job_id = await self.queue.get()
                job = self.jobs[job_id]
                job.status = 'processing'
                
                try:
                    # Execute synchronous generation in a thread pool to avoid blocking the event loop
                    result_bytes = await asyncio.to_thread(self._process_job, job)
                    job.result = result_bytes
                    job.status = 'done'
                except Exception as e:
                    job.error = str(e)
                    job.status = 'error'
                finally:
                    job.completed_at = datetime.utcnow()
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch worker loop: {e}")

    def _process_job(self, job: TTSJob) -> bytes:
        if job.model_type not in self.models:
            raise ValueError(f"Model type not found: {job.model_type}")
        
        model = self.models[job.model_type]
        
        # Determine generation arguments based on model type
        if job.model_type == 'multilingual':
            audio_tensor = model.generate(job.text, audio_prompt_path=job.voice_ref, lang=job.lang)
        elif job.model_type == 'turbo':
            audio_tensor = model.generate(job.text, audio_prompt_path=job.voice_ref)
        else: # Standard
            audio_tensor = model.generate(job.text, audio_prompt_path=job.voice_ref)
            
        # Convert tensor to wav bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, model.sr, format="wav")
        return buffer.getvalue()
