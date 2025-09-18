import os
import time
import json
import boto3
import requests
import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from google.cloud import translate_v3
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Directories
TRANSCRIPTS_DIRECTORY = "transcripts"
AUDIO_UPLOADS_DIRECTORY = "audio_uploads"
os.makedirs(TRANSCRIPTS_DIRECTORY, exist_ok=True)
os.makedirs(AUDIO_UPLOADS_DIRECTORY, exist_ok=True)

# Response Model
class AudioResponse(BaseModel):
    resp: str
    language: str

class AudioTranscriptionService:
    def __init__(self):
        self.aws_region = os.getenv('AWS_DEFAULT_REGION')
        self.s3_bucket = os.getenv('AWS_BUCKET')
        if not self.aws_region:
            raise ValueError("AWS_DEFAULT_REGION not set")
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET_NAME not set")
        
        self.transcribe_client = boto3.client('transcribe', region_name=self.aws_region)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
    
    async def upload_to_s3(self, file_path: str, file_name: str) -> str:
        """Upload audio file to S3 and return S3 URI"""
        try:
            s3_key = f"audio-uploads/{file_name}"
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key)
            s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
            print(f"Uploaded to S3: {s3_uri}")
            return s3_uri
        except Exception as e:
            raise Exception(f"S3 upload error: {e}")
    
    def download_transcription_file(self, uri: str, filename: str):
        """Download transcription file from URI"""
        try:
            response = requests.get(uri, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(TRANSCRIPTS_DIRECTORY, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath
        except Exception as e:
            raise Exception(f"Error downloading transcript: {e}")
    
    async def transcribe_audio_file(self, file_path: str, file_name: str, language: str) -> str:
        """Upload file to S3 and transcribe using AWS Transcribe"""
        try:
            # Upload to S3 first
            s3_uri = await self.upload_to_s3(file_path, file_name)
            
            # Start transcription job
            job_name = f"job-{int(time.time())}-{str(uuid.uuid4())[:8]}"
            
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                LanguageCode=language
            )
            
            # Wait for completion
            max_tries = 60
            while max_tries > 0:
                max_tries -= 1
                job = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                job_status = job['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status == 'COMPLETED':
                    transcription_uri = job['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    
                    # Download and parse transcript
                    filepath = self.download_transcription_file(
                        transcription_uri, 
                        f"{job_name}-output.json"
                    )
                    
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    transcript = data["results"]["transcripts"][0]["transcript"]
                    
                    # Clean up uploaded file from S3 (optional)
                    # self.s3_client.delete_object(Bucket=self.s3_bucket, Key=f"audio-uploads/{file_name}")
                    
                    return transcript
                    
                elif job_status == 'FAILED':
                    raise Exception(f"Transcription job failed: {job_name}")
                
                print(f"Waiting for transcription... Status: {job_status}")
                time.sleep(10)
            
            raise Exception("Transcription job timed out")
            
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

class TranslationService:
    def __init__(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not set")
        
        # Load from env var
        google_creds_b64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not google_creds_b64:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not set")

        creds_dict = json.loads(google_creds_b64)
        self.client = translate_v3.TranslationServiceClient.from_service_account_info(creds_dict)
        self.parent = f"projects/{self.project_id}/locations/global"
    
    def translate_text(self, text: str, target_language: str, source_language: str = None) -> str:
        """Translate text"""
        try:
            response = self.client.translate_text(
                contents=[text],
                target_language_code=target_language,
                parent=self.parent,
                mime_type="text/plain",
                source_language_code=source_language
            )
            return response.translations[0].translated_text
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")

class LLMService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )
    
    async def process_with_llm(self, text: str) -> str:
        """Process text with Gemini LLM using a default helpful prompt"""
        try:
            prompt = f"""
            Please analyze the following text and provide a helpful, informative response. 
            Summarize the key points and provide any relevant insights or recommendations based on the content.
            Keep your response clear, concise, and actionable.
            
            Text: {text}
            
            Response:
            """
            
            # Get response from LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            raise Exception(f"LLM processing error: {str(e)}")

class AudioPipelineOrchestrator:
    def __init__(self):
        self.transcription_service = AudioTranscriptionService()
        self.translation_service = TranslationService()
        self.llm_service = LLMService()
    
    def extract_language_code(self, language_with_region: str) -> str:
        """Extract language code from region-specific code (e.g., 'hi-IN' -> 'hi')"""
        return language_with_region.split('-')[0]
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to local directory"""
        try:
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(AUDIO_UPLOADS_DIRECTORY, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return file_path, unique_filename
        except Exception as e:
            raise Exception(f"File save error: {str(e)}")
    
    async def process_audio_file(self, file: UploadFile, source_language: str) -> AudioResponse:
        """Main pipeline: Audio File -> Upload -> Transcribe -> Translate -> LLM -> Translate back"""
        file_path = None
        try:
            # Step 1: Save uploaded file
            print(f"Saving uploaded file: {file.filename}")
            file_path, unique_filename = await self.save_uploaded_file(file)
            
            # Step 2: Transcribe audio
            print(f"Transcribing audio file: {unique_filename}")
            original_transcript = await self.transcription_service.transcribe_audio_file(
                file_path, unique_filename, source_language
            )
            print(f"Transcript: {original_transcript[:100]}...")
            
            # Step 3: Get language code for translation
            source_lang_code = self.extract_language_code(source_language)
            
            # Step 4: Translate to English (if not already English)
            if source_lang_code.lower() != 'en':
                print("Translating to English")
                english_text = self.translation_service.translate_text(
                    original_transcript,
                    target_language="en",
                    source_language=source_lang_code
                )
            else:
                english_text = original_transcript
            
            # Step 5: Process with Gemini LLM
            print("Processing with Gemini LLM")
            llm_response = await self.llm_service.process_with_llm(english_text)
            
            # Step 6: Translate back to original language (if not English)
            if source_lang_code.lower() != 'en':
                print("Translating back to original language")
                final_response = self.translation_service.translate_text(
                    llm_response,
                    target_language=source_lang_code,
                    source_language="en"
                )
            else:
                final_response = llm_response
            
            return AudioResponse(
                resp=final_response,
                language=source_lang_code
            )
            
        except Exception as e:
            raise Exception(f"Pipeline error: {str(e)}")
        finally:
            # Clean up local file
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up local file: {file_path}")
                except:
                    pass

# Initialize the orchestrator
pipeline = AudioPipelineOrchestrator()

@app.post("/process-audio/", response_model=AudioResponse)
async def process_audio(
    file: UploadFile = File(...),
    source_language: str = Form(...)
):
    """
    Process uploaded audio file: Audio -> Upload to S3 -> Transcribe -> Translate -> LLM -> Translate back
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - source_language: Language code (e.g., "hi-IN", "en-US", "es-ES")
    
    Returns: {"resp": "response text", "language": "language_code"}
    """
    
    # Validate file type
    allowed_types = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Validate file size (e.g., max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    # Reset file pointer
    await file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
        )
    
    try:
        result = await pipeline.process_audio_file(file, source_language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "aws_transcribe": "✓" if os.getenv('AWS_DEFAULT_REGION') else "✗",
            "s3_bucket": "✓" if os.getenv('S3_BUCKET_NAME') else "✗",
            "google_translate": "✓" if os.getenv('GOOGLE_CLOUD_PROJECT') else "✗",
            "gemini_llm": "✓" if os.getenv('GOOGLE_API_KEY') else "✗"
        }
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Multilingual Audio Processing API - File Upload",
        "endpoint": "/process-audio/",
        "method": "POST",
        "parameters": {
            "file": "Audio file (mp3, wav, m4a, etc.)",
            "source_language": "Language code (e.g., 'hi-IN', 'en-US')"
        },
        "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"],
        "max_file_size": "100MB",
        "response_format": {
            "resp": "AI generated response text",
            "language": "source language code"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)