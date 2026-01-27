"""
FastAPI application for the RAG Yacht Chatbot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import uvicorn
import logging
import json
import os
import re
import uuid
import requests  # For calling external APIs (OCR, chunking, embedding, Qdrant)
from datetime import datetime, timedelta
from config import Config
from agent import SimpleRAGChatbot
from agent_streaming import StreamingRAGChatbot
import prompts_url_creation  # URL creation prompts module
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Configure comprehensive logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # File handler for all logs
        logging.FileHandler(f'logs/mainapi_{datetime.now().strftime("%Y%m%d")}.log'),
        # Console handler for immediate feedback
        logging.StreamHandler()
    ]
)

# Set up specific loggers for different components
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger('agent')
streaming_logger = logging.getLogger('agent_streaming')
services_logger = logging.getLogger('services')

# Initialize FastAPI app
app = FastAPI(
    title="RAG Yacht Chatbot API",
    description="A comprehensive yacht industry chatbot powered by RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Request body size limits are configured at uvicorn startup level
# See the uvicorn.run() call at the bottom of this file

# Initialize the chatbots (will be created per-request with provider selection)
chatbot = None
streaming_chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize the default chatbots on startup"""
    global chatbot, streaming_chatbot
    try:
        logger.info("Initializing RAG Yacht Chatbot...")
        # Initialize with default provider
        chatbot = SimpleRAGChatbot()
        streaming_chatbot = StreamingRAGChatbot()
        logger.info("Chatbots initialized successfully!")
        
        # Create shared_conversations directory if it doesn't exist
        shared_conv_dir = "shared_conversations"
        os.makedirs(shared_conv_dir, exist_ok=True)
        logger.info(f"✅ Shared conversations directory ready: {shared_conv_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise e

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'tool'")
    content: Optional[str] = Field(None, description="Message content (can be None for tool calls)")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    name: Optional[str] = Field(None, description="Tool name (for tool messages)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls (for assistant messages with tool calls)")
    
    class Config:
        extra = "allow"  # Allow extra fields like tool_calls, name, etc.

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question about yachts (not required when share_conversation=true)")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation messages for context")
    provider: Optional[str] = Field(default="VLLM", description="LLM provider: 'VLLM', 'openai', 'gemini', 'openrouter', 'openrouter-grok-fast', or 'openrouter-claude'")
    accumulated_context: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Accumulated context from previous tool calls (can be string or object)")
    enable_exa_search: Optional[bool] = Field(default=True, description="Enable Exa web search")
    custom_system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to override default behavior (AI personality)")
    upload_custom_document: Optional[bool] = Field(default=False, description="If true, treat 'question' as document content to upload")
    document_title: Optional[str] = Field(default=None, description="Optional title for uploaded document (only used when upload_custom_document=true)")
    document_type: Optional[str] = Field(default="text", description="Document type: 'text', 'pdf', 'youtube', 'instagram', or 'url' (only used when upload_custom_document=true)")
    document_yacht_name: Optional[str] = Field(default=None, description="Optional yacht name (e.g., Karma, Jesma II)")
    document_url: Optional[str] = Field(default=None, description="Optional URL for youtube/instagram/url scraping (only used when upload_custom_document=true)")
    document_date: Optional[str] = Field(default=None, description="Optional date in YYYY-MM-DD format (only used when upload_custom_document=true)")
    share_conversation: Optional[bool] = Field(default=False, description="If true, share the conversation instead of chatting")
    share_title: Optional[str] = Field(default=None, description="Conversation title to share (only used when share_conversation=true)")
    share_history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history to share (only used when share_conversation=true)")
    share_context: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Accumulated context to share (only used when share_conversation=true)")
    share_timestamp: Optional[int] = Field(default=None, description="Original conversation timestamp (only used when share_conversation=true)")
    load_shared_conversation: Optional[bool] = Field(default=False, description="If true, load a shared conversation instead of chatting")
    share_id: Optional[str] = Field(default=None, description="Share ID to load (only used when load_shared_conversation=true)")
    submit_feedback: Optional[bool] = Field(default=False, description="If true, submit user feedback (vote/report) for a message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for feedback (only used when submit_feedback=true)")
    message_index: Optional[int] = Field(default=None, description="Message index for feedback (only used when submit_feedback=true)")
    vote: Optional[str] = Field(default=None, description="Vote type: 'thumbs_up' or 'thumbs_down' (only used when submit_feedback=true)")
    reported: Optional[bool] = Field(default=None, description="Whether message is reported (only used when submit_feedback=true)")
    timestamp: Optional[int] = Field(default=None, description="Feedback timestamp (only used when submit_feedback=true)")
    delete_documents: Optional[bool] = Field(default=False, description="If true, delete documents from Qdrant instead of chatting")
    delete_type: Optional[str] = Field(default=None, description="Document type to delete: 'news', 'doc', 'custom_pdf', 'custom_instagram', 'custom_url', 'custom_youtube', 'custom_text' (only used when delete_documents=true)")
    delete_identifier: Optional[str] = Field(default=None, description="URL or title to match for deletion (only used when delete_documents=true)")

    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the specifications of Princess V50?",
                "conversation_history": [],
                "provider": "VLLM",
                "accumulated_context": None,
                "enable_exa_search": True
            }
        }

class SemanticChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to chunk semantically")
    max_chunk_size: Optional[int] = Field(default=500, description="Maximum characters per chunk (guideline for LLM)")
    min_chunk_size: Optional[int] = Field(default=100, description="Minimum characters per chunk (guideline for LLM)")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Long text document that needs to be chunked semantically...",
                "max_chunk_size": 500,
                "min_chunk_size": 100
            }
        }

class SemanticChunkResponse(BaseModel):
    chunks: List[str] = Field(..., description="List of semantically chunked text segments")
    chunk_count: int = Field(..., description="Number of chunks created")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the chunking process")

class AudioTranscriptionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: Optional[str] = Field(default="wav", description="Audio format (wav, mp3, etc.)")
    guidance_prompt: Optional[str] = Field(default=None, description="Optional prompt to guide transcription (e.g., specific terms, context, or instructions)")
    
    class Config:
        schema_extra = {
            "example": {
                "audio_base64": "UklGRiQAAABXQVZFZm10...",
                "audio_format": "wav",
                "guidance_prompt": "This is a yacht industry conversation. Pay special attention to yacht model names like 'Princess V50' and technical terms."
            }
        }

class AudioTranscriptionResponse(BaseModel):
    transcription: str = Field(..., description="The transcribed text from the audio")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the transcription")

class YouTubeTranscriptionRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL to transcribe")
    
    class Config:
        schema_extra = {
            "example": {
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
        }

class YouTubeTranscriptionResponse(BaseModel):
    transcription: str = Field(..., description="The transcribed text from the YouTube video")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the transcription")

class SourceInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    vector_results: int = Field(description="Number of knowledge base results")
    yacht_results: int = Field(description="Number of yacht database results")
    builder_results: int = Field(description="Number of builder database results")
    country_results: int = Field(description="Number of country database results")
    global_results: int = Field(description="Number of global content results")
    model_results: int = Field(description="Number of yacht model results")
    yacht_detail_results: int = Field(description="Number of yacht detail results")
    model_detail_results: int = Field(description="Number of model detail results")

class SourceDisplayItem(BaseModel):
    """User-friendly source item for frontend display"""
    title: str = Field(description="Source title (user-friendly)")
    type: str = Field(description="Human-readable type (e.g. 'News Article', 'Yacht', 'YouTube Video')")
    icon: str = Field(description="Emoji or icon for visual representation")
    url: Optional[str] = Field(default=None, description="Link to source")
    date: Optional[str] = Field(default=None, description="Publication or creation date")
    snippet: str = Field(description="Short preview/summary text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (author, channel, etc.)")

class GraphData(BaseModel):
    type: str = Field(description="Graph type (line, bar, comparison)")
    title: str = Field(description="Graph title")
    data: Dict[str, Any] = Field(description="Graph data with labels and datasets")

class ChatResponse(BaseModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="The chatbot's answer")
    graphs: list = Field(default=[], description="Structured data for rendering graphs")
    sources: SourceInfo = Field(description="Information about sources used")
    sources_display: List[SourceDisplayItem] = Field(default=[], description="User-friendly source list for frontend display")
    total_sources: int = Field(description="Total number of sources consulted")
    context_length: int = Field(description="Length of context used")
    conversation_history: List[ChatMessage] = Field(default=[], description="Updated conversation history")
    accumulated_context: Optional[Dict[str, Any]] = Field(default=None, description="Accumulated context to send with next request")
    success: bool = Field(default=True, description="Whether the request was successful")

class ErrorResponse(BaseModel):
    error: str = Field(description="Error message")
    success: bool = Field(default=False, description="Whether the request was successful")

# Share conversation models
class ShareConversationRequest(BaseModel):
    title: str = Field(..., description="Conversation title")
    history: List[ChatMessage] = Field(..., description="Conversation history")
    context: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Accumulated context")
    timestamp: int = Field(..., description="Original conversation timestamp")

class ShareConversationResponse(BaseModel):
    share_id: str = Field(..., description="Unique share ID")
    share_url: str = Field(..., description="Shareable URL")
    expires_at: Optional[int] = Field(default=None, description="Expiration timestamp (optional)")

class SharedConversationResponse(BaseModel):
    id: str = Field(..., description="Share ID")
    title: str = Field(..., description="Conversation title")
    timestamp: int = Field(..., description="Original conversation timestamp")
    history: List[ChatMessage] = Field(..., description="Conversation history")
    context: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Accumulated context")
    created_at: int = Field(..., description="When the share was created")

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Yacht Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "status": "healthy",
        "service": "RAG Yacht Chatbot API"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the yacht expert chatbot
    
    This endpoint processes a user's question about yachts and returns a comprehensive answer
    based on multiple data sources including knowledge base, yacht database, builder database,
    and global content.
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        provider = request.provider or "VLLM"
        logger.info(f"Processing question: {request.question} (provider: {provider})")
        
        # Parse accumulated_context if it's a string
        accumulated_context = request.accumulated_context
        if accumulated_context and isinstance(accumulated_context, str):
            try:
                accumulated_context = json.loads(accumulated_context)
                logger.info("📦 Parsed accumulated_context from string to object")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse accumulated_context string: {e}")
                accumulated_context = None
        
        # Convert conversation history to the format expected by the chatbot
        conversation_history = []
        if request.conversation_history:
            for msg in request.conversation_history:
                msg_dict = {
                    'role': msg.role,
                    'content': msg.content if msg.content is not None else None
                }
                # Include tool_calls if present (for assistant messages with tool calls)
                if hasattr(msg, 'tool_calls') and msg.tool_calls is not None:
                    msg_dict['tool_calls'] = msg.tool_calls
                # Include name if present (for tool messages)
                if hasattr(msg, 'name') and msg.name is not None:
                    msg_dict['name'] = msg.name
                conversation_history.append(msg_dict)
        
        # Create chatbot instance with selected provider
        active_chatbot = SimpleRAGChatbot(provider=provider, enable_exa_search=request.enable_exa_search)
        
        # Get response from chatbot with accumulated context
        response = active_chatbot.ask(
            request.question, 
            conversation_history,
            accumulated_context=accumulated_context
        )
        
        # Build updated conversation history
        updated_history = []
        if request.conversation_history:
            updated_history.extend([ChatMessage(role=msg.role, content=msg.content) for msg in request.conversation_history])
        
        # Add current question and answer
        updated_history.append(ChatMessage(role="user", content=request.question))
        updated_history.append(ChatMessage(role="assistant", content=response["answer"]))
        
        # Convert to API response format
        api_response = ChatResponse(
            question=response["question"],
            answer=response["answer"],
            graphs=response.get("graphs", []),
            sources=SourceInfo(**response["sources"]),
            sources_display=response.get("sources_display", []),
            total_sources=response["total_sources"],
            context_length=response["context_length"],
            conversation_history=updated_history,
            accumulated_context=response.get("accumulated_context", None)
        )
        
        logger.info(f"Successfully processed question with {api_response.total_sources} sources")
        logger.info(f"📤 FINAL API RESPONSE - Answer (first 500 chars): {api_response.answer[:500]}")
        logger.info(f"📤 FINAL API RESPONSE - Total sources: {api_response.total_sources}, Context length: {api_response.context_length}")
        return api_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/chat/admin", response_class=HTMLResponse)
async def chat_admin_panel():
    """Temporary admin panel placeholder for testing."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <title>Admin Panel</title>
      </head>
      <body>
        <h1>Admin Panel</h1>
        <p>Admin interface coming soon.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Stream chat responses with real-time progress updates
    
    This endpoint uses Server-Sent Events (SSE) to stream:
    1. Progress updates during search ("Searching yachts...", "Found 5 results...")
    2. LLM response as it's being generated (word by word)
    
    Special mode: Document Upload
    - Set upload_custom_document=true to upload a document instead of chatting
    - The 'question' field becomes the document content (plain text only)
    - Use 'document_title' for optional title
    
    Event types:
    - status: Search progress updates
    - answer_start: Answer generation begins
    - answer_chunk: Partial answer content (stream this to user)
    - answer_complete: Full answer ready
    - complete: Everything finished with metadata
    - error: Error occurred
    """
    if streaming_chatbot is None:
        raise HTTPException(status_code=503, detail="Streaming chatbot not initialized")
    
    # Read raw body to bypass default size limits (allows up to 100MB)
    # This is necessary for large base64-encoded PDF uploads
    try:
        body_bytes = await request.body()
        # Parse JSON manually to avoid Pydantic's size restrictions
        request_data = json.loads(body_bytes.decode('utf-8'))
        
        # If sharing conversation, question field is not required - use placeholder
        if request_data.get('share_conversation'):
            if not request_data.get('question') or request_data.get('question') == '':
                request_data['question'] = 'share_conversation_placeholder'
        
        # Create ChatRequest object from parsed data
        request_obj = ChatRequest(**request_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")
    
    async def event_generator():
        """Generate Server-Sent Events with immediate flushing"""
        import asyncio
        import uuid
        try:
            # DOCUMENT UPLOAD MODE
            if request_obj.upload_custom_document:
                logger.info(f"📥 Document upload mode activated via /chat/stream")
                try:
                    title = request_obj.document_title
                    document_type = request_obj.document_type or "text"
                    manufacturer = None  # Not applicable for custom documents
                    model_name = None  # Not applicable for custom documents
                    yacht_name = request_obj.document_yacht_name
                    document_url = request_obj.document_url  # Store the URL (for youtube/instagram)
                    document_date = request_obj.document_date  # Store the date in YYYY-MM-DD format
                    extracted_text = ""
                    
                    # Instagram-specific metadata (for context prefixing)
                    instagram_caption = None
                    instagram_owner = None
                    
                    # Handle PDF upload
                    if document_type == "pdf":
                        logger.info(f"📄 Processing PDF document...")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing PDF...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        
                        try:
                            import base64
                            import fitz  # PyMuPDF
                            
                            # Decode base64 PDF
                            pdf_base64 = request_obj.question
                            pdf_bytes = base64.b64decode(pdf_base64)
                            
                            # Open PDF
                            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                            total_pages = len(pdf)
                            logger.info(f"📄 PDF has {total_pages} pages")
                            
                            yield f"data: {json.dumps({'type': 'status', 'message': f'OCR processing {total_pages} pages...'}, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                            
                            # OCR each page - use LLM URL from config
                            from config import Config as ConfigClass
                            ocr_url = ConfigClass.LLM_URL
                            page_texts = []
                            
                            # Check if OCR service is available before processing
                            # Try a simple connection test (health endpoint may not exist, so we'll catch errors on first request)
                            logger.info(f"🔍 Checking OCR service availability at {ocr_url}...")
                            
                            for page_num in range(total_pages):
                                logger.info(f"🔍 OCR processing page {page_num + 1}/{total_pages}...")
                                yield f"data: {json.dumps({'type': 'status', 'message': f'OCR page {page_num + 1}/{total_pages}...'}, ensure_ascii=True)}\n\n"
                                await asyncio.sleep(0.01)
                                
                                page = pdf[page_num]
                                pix = page.get_pixmap(dpi=150)
                                img_bytes = pix.tobytes("png")
                                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                
                                # Call OCR API
                                ocr_payload = {
                                    "model": ConfigClass.LLM_MODEL,
                                    "messages": [{
                                        "role": "user",
                                        "content": [
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                                            {"type": "text", "text": "Extract all text from this document image. Provide the complete text content exactly as it appears."}
                                        ]
                                    }],
                                    "max_tokens": 4000
                                }
                                
                                try:
                                    ocr_response = requests.post(ocr_url, json=ocr_payload, timeout=120)
                                    if ocr_response.status_code != 200:
                                        raise Exception(f"OCR failed for page {page_num + 1}: {ocr_response.text}")
                                except requests.exceptions.ConnectionError as e:
                                    raise Exception(f"Failed to connect to OCR service at {ocr_url}. Please ensure the service is running. Error: {str(e)}")
                                except requests.exceptions.Timeout as e:
                                    raise Exception(f"OCR request timed out for page {page_num + 1}. The PDF might be too large or the service is overloaded. Error: {str(e)}")
                                except requests.exceptions.RequestException as e:
                                    raise Exception(f"OCR request failed for page {page_num + 1}: {str(e)}")
                                
                                ocr_result = ocr_response.json()
                                page_text = ocr_result['choices'][0]['message']['content']
                                page_texts.append(page_text)
                                logger.info(f"✅ Page {page_num + 1} OCR complete: {len(page_text)} chars")
                            
                            pdf.close()
                            extracted_text = "\n\n".join(page_texts)
                            logger.info(f"✅ PDF OCR complete: {len(extracted_text)} total characters")
                            
                        except Exception as e:
                            logger.error(f"❌ PDF processing failed: {e}")
                            raise Exception(f"PDF processing failed: {str(e)}")
                    
                    # Handle Instagram Reel transcription
                    elif document_type == "instagram":
                        logger.info(f"📱 Processing Instagram Reel...")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Transcribing Instagram Reel...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        
                        try:
                            from apify_client import ApifyClient
                            from config import Config as ConfigClass
                            
                            # Get API keys from config
                            apify_token = ConfigClass.APIFY_API_TOKEN
                            openai_key = ConfigClass.OPENAI_API_KEY
                            
                            if not apify_token:
                                raise ValueError("APIFY_API_TOKEN not configured. Please set it in config.py or environment variable.")
                            if not openai_key:
                                raise ValueError("OPENAI_API_KEY not configured. Please set it in config.py or environment variable.")
                            
                            # Prefer document_url if provided, otherwise use question field
                            instagram_url = (document_url or request_obj.question).strip()
                            if not instagram_url:
                                raise ValueError("Instagram Reel URL is required. Provide it in 'document_url' or 'question' field.")
                            logger.info(f"📱 Instagram URL: {instagram_url}")
                            
                            # Initialize Apify client
                            client = ApifyClient(apify_token)
                            
                            # Prepare Actor input
                            run_input = {
                                "instagramUrl": instagram_url,
                                "openaiApiKey": openai_key,
                                "task": "transcription",
                                "model": "gpt-4o-mini-transcribe",
                                "response_format": "json"
                            }
                            
                            logger.info(f"🚀 Starting Apify Actor for Instagram transcription...")
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Downloading Instagram Reel...'}, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                            
                            # Run the Actor and wait for it to finish
                            run = client.actor("linen_snack/instagram-videos-transcipt-subtitles-and-translate").call(run_input=run_input)
                            
                            logger.info(f"✅ Apify Actor completed. Fetching results...")
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Transcription complete. Processing text...'}, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                            
                            # Fetch results from the dataset
                            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                            
                            if not items:
                                raise ValueError("No transcription results returned from Apify Actor")
                            
                            # Extract the transcribed text
                            result_item = items[0]
                            
                            # DEBUG: Log full response structure to check for captions
                            logger.info(f"🔍 Full API response keys: {list(result_item.keys())}")
                            if isinstance(result_item.get('result'), dict):
                                logger.info(f"🔍 Result object keys: {list(result_item.get('result', {}).keys())}")
                            
                            transcription_result = result_item.get('result', {})
                            extracted_text = transcription_result.get('text', '')
                            
                            # Try to extract caption if available
                            caption = None
                            # Check various possible caption fields
                            caption_fields = ['caption', 'captions', 'description', 'post_caption', 
                                            'user_caption', 'original_caption', 'caption_text', 
                                            'edge_media_to_caption', 'text_caption']
                            
                            # Check in result object
                            if isinstance(transcription_result, dict):
                                for field in caption_fields:
                                    if field in transcription_result and transcription_result[field]:
                                        caption = transcription_result[field]
                                        logger.info(f"✅ Found caption in field '{field}': {str(caption)[:200]}...")
                                        break
                            
                            # Check in top-level result_item
                            if not caption:
                                for field in caption_fields:
                                    if field in result_item and result_item[field]:
                                        caption = result_item[field]
                                        logger.info(f"✅ Found caption in top-level field '{field}': {str(caption)[:200]}...")
                                        break
                            
                            if not extracted_text:
                                raise ValueError("Transcription returned empty text")
                            
                            # If caption found, append it to the extracted text and set title
                            if caption:
                                caption_text = str(caption).strip()
                                instagram_caption = caption_text  # Store for context prefixing
                                if caption_text and caption_text not in extracted_text:
                                    # Prepend caption to transcript
                                    extracted_text = f"[Caption: {caption_text}]\n\n[Transcript: {extracted_text}]"
                                    logger.info(f"✅ Added caption to transcript: {len(caption_text)} characters")
                                    
                                    # Use caption as title if no title was provided
                                    if not title:
                                        # Use first line or first 100 chars of caption as title
                                        if '\n' in caption_text:
                                            title = caption_text.split('\n')[0][:100]
                                        else:
                                            title = caption_text[:100] if len(caption_text) <= 100 else caption_text[:97] + "..."
                                        logger.info(f"📝 Reel title set from caption: {title}")
                            
                            logger.info(f"✅ Instagram transcription complete: {len(extracted_text)} characters")
                            
                            # Always try to fetch reel metadata (caption, title, etc.) using Instagram Reel Scraper
                            # This ensures we get title even if caption wasn't found in transcription response
                            if not caption or not title:
                                logger.info("🔍 Fetching reel metadata from Instagram Reel Scraper...")
                                try:
                                    # Use Apify's Instagram Reel Scraper to get post metadata including captions
                                    # The username array can contain reel URLs directly
                                    reel_scraper_input = {
                                        "username": [instagram_url],
                                        "resultsLimit": 1
                                    }
                                    
                                    reel_scraper_run = client.actor("apify/instagram-reel-scraper").call(run_input=reel_scraper_input)
                                    reel_scraper_items = list(client.dataset(reel_scraper_run["defaultDatasetId"]).iterate_items())
                                    
                                    if reel_scraper_items:
                                        reel_data = reel_scraper_items[0]
                                        
                                        # Extract reel metadata for title and other info
                                        reel_shortcode = reel_data.get('shortCode', '')
                                        reel_owner = reel_data.get('ownerUsername', '')
                                        
                                        # Store owner for context prefixing
                                        if reel_owner:
                                            instagram_owner = reel_owner
                                        
                                        # Set title if not already set
                                        if not title:
                                            if 'caption' in reel_data and reel_data['caption']:
                                                # Use first line or first 100 chars of caption as title
                                                caption_for_title = str(reel_data['caption']).strip()
                                                if '\n' in caption_for_title:
                                                    title = caption_for_title.split('\n')[0][:100]
                                                else:
                                                    title = caption_for_title[:100] if len(caption_for_title) <= 100 else caption_for_title[:97] + "..."
                                            elif reel_owner and reel_shortcode:
                                                title = f"Instagram Reel by @{reel_owner} ({reel_shortcode})"
                                            elif reel_shortcode:
                                                title = f"Instagram Reel ({reel_shortcode})"
                                            else:
                                                title = "Instagram Reel"
                                        
                                        # Check for caption field if we don't have one yet
                                        if not caption and 'caption' in reel_data and reel_data['caption']:
                                            caption = reel_data['caption']
                                            logger.info(f"✅ Found caption via Instagram Reel Scraper: {str(caption)[:200]}...")
                                            
                                            caption_text = str(caption).strip()
                                            instagram_caption = caption_text  # Store for context prefixing
                                            if caption_text and caption_text not in extracted_text:
                                                extracted_text = f"[Caption: {caption_text}]\n\n[Transcript: {extracted_text}]"
                                                logger.info(f"✅ Added caption to transcript: {len(caption_text)} characters")
                                        
                                        logger.info(f"📝 Reel title set to: {title}")
                                        if reel_owner:
                                            logger.info(f"👤 Reel owner: @{reel_owner}")
                                        if reel_shortcode:
                                            logger.info(f"🔗 Reel shortcode: {reel_shortcode}")
                                    
                                except Exception as scraper_error:
                                    logger.warning(f"⚠️ Could not fetch reel metadata via Instagram Reel Scraper: {str(scraper_error)[:200]}")
                                    # Set default title if we still don't have one
                                    if not title:
                                        title = "Instagram Reel"
                                    # Continue without caption - transcription is still available
                            
                        except Exception as e:
                            logger.error(f"❌ Instagram transcription failed: {e}")
                            raise Exception(f"Instagram transcription failed: {str(e)}")
                    
                    # Handle plain text
                    elif document_type == "text":
                        extracted_text = request_obj.question.strip()
                        logger.info(f"📝 Processing text document: {len(extracted_text)} chars, title={title}")
                    
                    # Handle YouTube URL transcription
                    elif document_type == "youtube":
                        youtube_url = request_obj.question.strip()
                        logger.info(f"🎬 Processing YouTube URL: {youtube_url}")
                        
                        # Validate YouTube URL
                        if not youtube_url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
                            raise ValueError("Invalid YouTube URL format")
                        
                        yield f"data: {json.dumps({'type': 'status', 'message': '🎬 Fetching YouTube transcript...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        
                        # Use youtube-transcript-api to fetch transcript directly
                        from youtube_transcript_api import YouTubeTranscriptApi
                        from youtube_transcript_api.proxies import WebshareProxyConfig
                        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked
                        
                        # Extract video ID from URL
                        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
                        if not video_id_match:
                            raise ValueError("Could not extract video ID from YouTube URL")
                        
                        video_id = video_id_match.group(1)
                        logger.info(f"📹 Extracted video ID: {video_id}")
                        
                        # Create API instance with proxy to bypass IP blocking
                        api = YouTubeTranscriptApi(
                            proxy_config=WebshareProxyConfig(
                                proxy_username="amxukcgo",
                                proxy_password="q0nxvwzetxjz",
                            )
                        )
                        
                        transcript_text_parts = None
                        try:
                            # Fetch transcript (tries English first)
                            fetched_transcript = api.fetch(video_id, languages=['en'])
                            # Extract text from FetchedTranscriptSnippet objects
                            transcript_text_parts = [snippet.text for snippet in fetched_transcript]
                            logger.info(f"✅ Found English transcript with {len(transcript_text_parts)} entries")
                        except NoTranscriptFound:
                            try:
                                # Try any available language
                                logger.info("⚠️ No English transcript, trying other languages...")
                                available_transcripts = api.list(video_id)
                                for transcript in available_transcripts:
                                    try:
                                        fetched_transcript = transcript.fetch()
                                        transcript_text_parts = [snippet.text for snippet in fetched_transcript]
                                        logger.info(f"✅ Found transcript in language: {transcript.language}")
                                        break
                                    except:
                                        continue
                                
                                if not transcript_text_parts:
                                    raise NoTranscriptFound("No transcript found in any language")
                            except Exception as e:
                                raise Exception(f"No transcript available for this video: {str(e)}")
                        except TranscriptsDisabled:
                            raise Exception("Transcripts are disabled for this video")
                        except VideoUnavailable:
                            raise Exception("Video is unavailable or private")
                        except IpBlocked:
                            logger.warning(f"⚠️ YouTube IP blocked, falling back to Gemini...")
                            # Fallback to Gemini when IP is blocked
                            yield f"data: {json.dumps({'type': 'status', 'message': '⚠️ Direct access blocked, using Gemini AI as fallback...'}, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                            
                            # Use Gemini as fallback
                            from config import Config
                            config = Config()
                            
                            model_id = "gemini-3-flash-preview"
                            api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
                            
                            user_prompt = f"""Please fetch and transcribe the audio from this YouTube video: {youtube_url}

Requirements:
- Use Google Search tool to access the video
- Extract the full transcript
- Return ONLY the clean text transcript
- Do NOT include timestamps
- Do NOT include any commentary or additional text
- Just the pure transcript text"""

                            payload = {
                                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                                "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}},
                                "tools": [{"googleSearch": {}}]
                            }
                            
                            url = f"{api_endpoint}?key={config.GEMINI_API_KEY}"
                            transcription_response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=600)
                            
                            if transcription_response.status_code != 200:
                                raise Exception(f"Gemini fallback also failed: {transcription_response.status_code}")
                            
                            response_json = transcription_response.json()
                            transcription = ""
                            
                            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                                candidate = response_json["candidates"][0]
                                if "content" in candidate:
                                    content = candidate["content"]
                                    if "parts" in content:
                                        for part in content["parts"]:
                                            if "text" in part:
                                                transcription = part["text"].strip()
                                                break
                                    elif "text" in content:
                                        transcription = content["text"].strip()
                            
                            if not transcription or len(transcription) < 100:
                                raise Exception("Gemini fallback returned invalid or empty transcript")
                            
                            # Check for error messages
                            error_indicators = ["I am unable to", "I cannot", "cannot fetch", "unable to fetch"]
                            if any(indicator in transcription.lower() for indicator in error_indicators):
                                raise Exception(f"Both YouTube API and Gemini failed. Gemini returned: {transcription[:200]}")
                            
                            extracted_text = transcription
                            logger.info(f"✅ Gemini fallback transcription complete: {len(extracted_text)} chars")
                        except Exception as e:
                            if "IpBlocked" not in str(type(e).__name__):
                                raise Exception(f"Failed to fetch transcript: {str(e)}")
                            else:
                                raise
                        
                        # Only combine transcript entries if we got them from YouTube API (not Gemini fallback)
                        if 'transcript_text_parts' in locals() and transcript_text_parts:
                            # Combine transcript text parts into clean text
                            extracted_text = " ".join(transcript_text_parts)
                        
                        # Clean up common transcript artifacts
                        extracted_text = extracted_text.replace('\n', ' ')
                        extracted_text = re.sub(r'\s+', ' ', extracted_text)  # Remove extra whitespace
                        extracted_text = extracted_text.strip()
                        
                        logger.info(f"✅ YouTube transcription complete: {len(extracted_text)} chars")
                        logger.info(f"📝 TRANSCRIPT PREVIEW (first 500 chars):\n{extracted_text[:500]}")
                        logger.info(f"📝 TRANSCRIPT PREVIEW (last 500 chars):\n{extracted_text[-500:]}")
                        yield f"data: {json.dumps({'type': 'status', 'message': f'✅ Transcription complete! ({len(extracted_text)} chars) - Now storing...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                    
                    # Handle custom URL scraping with Gemini
                    elif document_type == "url":
                        custom_url = request_obj.question.strip()
                        logger.info(f"🌐 Processing custom URL: {custom_url}")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Scraping URL with Gemini...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        
                        # Validate URL
                        if not custom_url.startswith(("http://", "https://")):
                            raise ValueError("Invalid URL format. Must start with http:// or https://")
                        
                        try:
                            # Use Gemini 3 Pro with urlContext and googleSearch tools
                            from config import Config as ConfigClass
                            api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:streamGenerateContent"
                            
                            prompt = f"""You are analyzing the content from this URL: {custom_url}

Please extract and provide:

1. **ARTICLE DATE**: If you find a publication date, last updated date, or article date on this page, extract it and format it EXACTLY as YYYY-MM-DD (e.g., 2024-12-12). If no date is found, respond with "NO_DATE_FOUND".

2. **CONTENT**: A comprehensive transcript of all useful content from this page. Focus on:
   - Main text content, articles, and descriptions
   - Important facts, data, and information
   - Technical specifications if present
   - Any general knowledge that would be valuable in a vector database
   - Exclude navigation menus, footers, ads, and boilerplate content

Format your response EXACTLY as follows:
```
DATE: YYYY-MM-DD or NO_DATE_FOUND
---
[Content here]
```

Provide the content in a clean, well-structured format that's suitable for storage and retrieval."""
                            
                            payload = {
                                "contents": [
                                    {
                                        "role": "user",
                                        "parts": [
                                            {
                                                "text": prompt
                                            }
                                        ]
                                    }
                                ],
                                "generationConfig": {
                                    "thinkingConfig": {
                                        "thinkingLevel": "HIGH"
                                    }
                                },
                                "tools": [
                                    {
                                        "urlContext": {}
                                    },
                                    {
                                        "googleSearch": {}
                                    }
                                ]
                            }
                            
                            url = f"{api_endpoint}?key={ConfigClass.GEMINI_API_KEY}"
                            logger.info(f"🤖 Calling Gemini 3 Pro with urlContext and googleSearch tools...")
                            
                            scrape_response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=600, stream=True)
                            
                            if scrape_response.status_code != 200:
                                logger.error(f"❌ Gemini API error: {scrape_response.status_code} - {scrape_response.text}")
                                raise Exception(f"Gemini URL scraping failed: {scrape_response.text}")
                            
                            # Parse streaming response - Gemini streams JSON chunks
                            extracted_text = ""
                            buffer = ""
                            
                            for line in scrape_response.iter_lines():
                                if line:
                                    line_str = line.decode('utf-8')
                                    buffer += line_str
                                    
                                    # Try to parse complete JSON objects from buffer
                                    try:
                                        # Gemini streams as JSON array of chunks
                                        json_data = json.loads(buffer)
                                        
                                        # Check if it's an array of responses
                                        if isinstance(json_data, list):
                                            for item in json_data:
                                                if 'candidates' in item:
                                                    for candidate in item['candidates']:
                                                        if 'content' in candidate and 'parts' in candidate['content']:
                                                            for part in candidate['content']['parts']:
                                                                if 'text' in part:
                                                                    extracted_text += part['text']
                                        # Or single response object
                                        elif 'candidates' in json_data:
                                            for candidate in json_data['candidates']:
                                                if 'content' in candidate and 'parts' in candidate['content']:
                                                    for part in candidate['content']['parts']:
                                                        if 'text' in part:
                                                            extracted_text += part['text']
                                        
                                        # Clear buffer after successful parse
                                        buffer = ""
                                    except json.JSONDecodeError:
                                        # Keep buffering until we have complete JSON
                                        continue
                            
                            # Try parsing any remaining buffer
                            if buffer:
                                try:
                                    json_data = json.loads(buffer)
                                    if isinstance(json_data, list):
                                        for item in json_data:
                                            if 'candidates' in item:
                                                for candidate in item['candidates']:
                                                    if 'content' in candidate and 'parts' in candidate['content']:
                                                        for part in candidate['content']['parts']:
                                                            if 'text' in part:
                                                                extracted_text += part['text']
                                    elif 'candidates' in json_data:
                                        for candidate in json_data['candidates']:
                                            if 'content' in candidate and 'parts' in candidate['content']:
                                                for part in candidate['content']['parts']:
                                                    if 'text' in part:
                                                        extracted_text += part['text']
                                except json.JSONDecodeError as e:
                                    logger.warning(f"⚠️ Failed to parse remaining buffer: {e}")
                            
                            logger.info(f"📊 Extracted text length: {len(extracted_text)}")
                            
                            if not extracted_text.strip():
                                logger.error(f"❌ No text extracted from Gemini response")
                                # Log a sample of the response for debugging
                                logger.error(f"❌ Buffer sample: {buffer[:500]}")
                                raise Exception("Gemini returned empty content from URL")
                            
                            # Parse date from Gemini response (format: "DATE: YYYY-MM-DD\n---\n[content]")
                            gemini_extracted_date = None
                            content_without_date = extracted_text
                            
                            # Look for DATE: pattern at the start
                            if extracted_text.startswith("DATE:"):
                                lines = extracted_text.split('\n', 2)  # Split into max 3 parts
                                if len(lines) >= 3 and lines[1].strip() == "---":
                                    # Extract date from first line
                                    date_str = lines[0].replace("DATE:", "").strip()
                                    content_without_date = lines[2]  # Content after ---
                                    if date_str != "NO_DATE_FOUND" and len(date_str) == 10:  # YYYY-MM-DD is 10 chars
                                        # Validate date format (YYYY-MM-DD)
                                        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                                            gemini_extracted_date = date_str
                                            logger.info(f"📅 Gemini extracted date from page: {gemini_extracted_date}")
                                        else:
                                            logger.warning(f"⚠️ Gemini returned invalid date format: {date_str}")
                            
                            # Use user-provided date first, fallback to Gemini-extracted date
                            if document_date:
                                # Validate user-provided date format
                                if re.match(r'^\d{4}-\d{2}-\d{2}$', document_date):
                                    logger.info(f"📅 Using user-provided date: {document_date}")
                                else:
                                    logger.warning(f"⚠️ User-provided date format invalid: {document_date}, trying to parse...")
                                    # Try to parse common date formats and convert to YYYY-MM-DD
                                    try:
                                        # Try common formats
                                        for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                                            try:
                                                parsed_date = datetime.strptime(document_date, fmt)
                                                document_date = parsed_date.strftime('%Y-%m-%d')
                                                logger.info(f"📅 Converted user date to YYYY-MM-DD: {document_date}")
                                                break
                                            except ValueError:
                                                continue
                                    except Exception as e:
                                        logger.warning(f"⚠️ Could not parse user date: {e}")
                                        document_date = None
                            elif gemini_extracted_date:
                                document_date = gemini_extracted_date
                                logger.info(f"📅 Using Gemini-extracted date: {document_date}")
                            else:
                                logger.info(f"⚠️ No date found (user input or page extraction)")
                            
                            # Use the cleaned content (without the DATE: header)
                            extracted_text = content_without_date.strip()
                            
                            logger.info(f"✅ URL scraping complete: {len(extracted_text)} chars")
                            logger.info(f"📝 CONTENT PREVIEW (first 500 chars):\n{extracted_text[:500]}")
                            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ Scraped {len(extracted_text)} characters from URL - Now storing...'}, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                            
                        except Exception as e:
                            logger.error(f"❌ URL scraping failed: {e}")
                            raise Exception(f"URL scraping failed: {str(e)}")
                    
                    else:
                        raise ValueError(f"Invalid document_type: {document_type}. Must be 'text', 'pdf', 'youtube', 'instagram', or 'url'")
                    
                    if not extracted_text.strip():
                        raise ValueError("Document content is empty after processing")
                    
                    # Send status update
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Chunking document...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Step 1: Chunk text
                    chunk_url = "http://localhost:8002/chunk"
                    chunk_payload = {"text": extracted_text, "algorithm": "word-count"}
                    chunk_response = requests.post(chunk_url, json=chunk_payload, timeout=120)
                    
                    if chunk_response.status_code != 200:
                        raise Exception(f"Chunking failed: {chunk_response.text}")
                    
                    chunk_result = chunk_response.json()
                    chunks = chunk_result['chunks']
                    logger.info(f"✅ Created {len(chunks)} chunks")
                    
                    # Add context prefix to chunks if this is a YouTube transcript
                    if document_type == "youtube":
                        # Build context prefix
                        context_parts = ["(context: this is one chunk out of a transcript of a video"]
                        
                        if title:
                            context_parts.append(f" with the title: {title}")
                        
                        context_parts.append(")")
                        
                        # Add optional metadata fields
                        metadata_parts = []
                        if manufacturer:
                            metadata_parts.append(f"Manufacturer: {manufacturer}")
                        if model_name:
                            metadata_parts.append(f"Model name: {model_name}")
                        if yacht_name:
                            metadata_parts.append(f"Yacht name: {yacht_name}")
                        
                        if metadata_parts:
                            context_parts.append(" | " + " | ".join(metadata_parts))
                        
                        context_prefix = " ".join(context_parts) + "\n\n"
                        
                        # Add prefix to each chunk
                        chunks = [context_prefix + chunk for chunk in chunks]
                        logger.info(f"✅ Added context prefix to {len(chunks)} chunks")
                        logger.debug(f"📝 Prefix example: {context_prefix[:200]}")
                    
                    # Add context prefix to chunks if this is an Instagram Reel
                    elif document_type == "instagram":
                        # Build context prefix with caption, owner, and URL
                        context_parts = ["(context: this is one chunk from an Instagram Reel"]
                        
                        # Build context prefix with metadata
                        metadata_parts = []
                        if instagram_caption:
                            # Use first line or first 150 chars of caption
                            caption_preview = instagram_caption.split('\n')[0] if '\n' in instagram_caption else instagram_caption
                            caption_preview = caption_preview[:150] + "..." if len(caption_preview) > 150 else caption_preview
                            metadata_parts.append(f"Caption: {caption_preview}")
                        if instagram_owner:
                            metadata_parts.append(f"Owner: @{instagram_owner}")
                        if document_url:
                            metadata_parts.append(f"URL: {document_url}")
                        if title:
                            metadata_parts.append(f"Title: {title}")
                        
                        context_parts.append(")")
                        if metadata_parts:
                            context_parts.append(" | " + " | ".join(metadata_parts))
                        
                        context_prefix = " ".join(context_parts) + "\n\n"
                        
                        # Add prefix to each chunk
                        chunks = [context_prefix + chunk for chunk in chunks]
                        logger.info(f"✅ Added Instagram context prefix to {len(chunks)} chunks")
                        logger.debug(f"📝 Prefix example: {context_prefix[:200]}")
                    else:
                        # For all other document types, prepend title to each chunk if provided
                        if title:
                            chunks = [f"{title}\n\n{chunk}" for chunk in chunks]
                            logger.info(f"✅ Prefixed title to {len(chunks)} chunks")
                    
                    # Send status update
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Created {len(chunks)} chunks. Generating embeddings...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Step 2: Generate embeddings
                    embed_url = "http://localhost:8080/embed"
                    embed_payload = {"texts": chunks}
                    embed_response = requests.post(embed_url, json=embed_payload, timeout=120)
                    
                    if embed_response.status_code != 200:
                        raise Exception(f"Embedding failed: {embed_response.text}")
                    
                    embed_result = embed_response.json()
                    embeddings = embed_result['embeddings']
                    logger.info(f"✅ Generated {len(embeddings)} embeddings")
                    
                    # Step 2.5: Generate sparse vectors for hybrid search
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating sparse vectors...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    if chatbot is None:
                        logger.warning("⚠️ Chatbot not initialized, creating temporary instance for sparse vectors...")
                        from agent import SimpleRAGChatbot
                        temp_chatbot = SimpleRAGChatbot()
                        sparse_service = temp_chatbot.vector_search_service.sparse_service
                    else:
                        sparse_service = chatbot.vector_search_service.sparse_service
                    
                    sparse_vectors = []
                    for chunk in chunks:
                        sparse_vec_dict = sparse_service.create_sparse_vector(chunk)
                        # Convert from {token_id: weight} dict to Qdrant format {indices: [...], values: [...]}
                        if sparse_vec_dict:
                            indices = [int(k) for k in sparse_vec_dict.keys()]
                            values = [float(v) for v in sparse_vec_dict.values()]
                            sparse_vec = {"indices": indices, "values": values}
                        else:
                            sparse_vec = {"indices": [], "values": []}
                        sparse_vectors.append(sparse_vec)
                    logger.info(f"✅ Generated {len(sparse_vectors)} sparse vectors")
                    
                    # Send status update
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Generated embeddings. Inserting into database...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Step 3: Insert into Qdrant
                    document_id = str(uuid.uuid4())
                    points = []
                    
                    for i, (chunk, embedding, sparse_vec) in enumerate(zip(chunks, embeddings, sparse_vectors)):
                        point = {
                            "id": str(uuid.uuid4()),
                            "vector": {
                                "content": embedding  # Dense vector
                            },
                            "sparse_vector": {
                                "content_sparse": sparse_vec  # Sparse vector for hybrid search
                            },
                            "payload": {
                                "text": chunk,  # Title already included in chunk text
                                "type": "custom",
                                "document_id": document_id,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        }
                        if title:
                            point["payload"]["title"] = title
                        if manufacturer:
                            point["payload"]["manufacturer"] = manufacturer
                        if model_name:
                            point["payload"]["model_name"] = model_name
                        if yacht_name:
                            point["payload"]["yacht_name"] = yacht_name
                        if document_type:
                            point["payload"]["document_type"] = document_type
                        if document_url:
                            point["payload"]["url"] = document_url
                        if document_date:
                            point["payload"]["date"] = document_date
                        points.append(point)
                    
                    # Use collection from config (qwen_4b_sparse)
                    from config import Config as ConfigClass
                    collection_name = ConfigClass.QDRANT_COLLECTION
                    qdrant_url = f"http://localhost:6333/collections/{collection_name}/points"
                    qdrant_payload = {"points": points}
                    qdrant_response = requests.put(qdrant_url, json=qdrant_payload, timeout=60)
                    
                    if qdrant_response.status_code != 200:
                        raise Exception(f"Qdrant insertion failed: {qdrant_response.text}")
                    
                    logger.info(f"✅ Successfully inserted {len(points)} points into Qdrant")
                    if document_url:
                        logger.info(f"📎 Document URL stored: {document_url}")
                    if document_date:
                        logger.info(f"📅 Document date stored: {document_date}")
                    
                    # Send success response
                    success_data = {
                        "type": "upload_complete",
                        "success": True,
                        "message": "Document uploaded successfully!",
                        "chunks_created": len(chunks),
                        "points_inserted": len(points),
                        "document_id": document_id,
                        "metadata": {
                            "title": title,
                            "total_characters": len(extracted_text),
                            "chunking_algorithm": "word-count"
                        }
                    }
                    yield f"data: {json.dumps(success_data, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Document upload failed: {e}", exc_info=True)
                    error_event = json.dumps({
                        "type": "error",
                        "message": f"Upload failed: {str(e)}"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
                    return
            
            # SHARE CONVERSATION MODE
            if request_obj.share_conversation:
                logger.info(f"🔗 Share conversation mode activated via /chat/stream")
                try:
                    # Validate share data
                    if not request_obj.share_history or len(request_obj.share_history) == 0:
                        raise ValueError("Cannot share empty conversation")
                    
                    if len(request_obj.share_history) > 100:
                        raise ValueError("Conversation too large (max 100 messages)")
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Creating share link...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Generate unique share ID
                    share_id = str(uuid.uuid4())
                    
                    # Create share data
                    share_data = {
                        "id": share_id,
                        "title": request_obj.share_title or "Shared Conversation",
                        "timestamp": request_obj.share_timestamp or int(datetime.now().timestamp() * 1000),
                        "history": [msg.dict() if hasattr(msg, 'dict') else msg for msg in request_obj.share_history],
                        "context": request_obj.share_context,
                        "created_at": int(datetime.now().timestamp() * 1000),
                        "expires_at": int((datetime.now() + timedelta(days=30)).timestamp() * 1000)  # 30 days expiration
                    }
                    
                    # Save to file
                    shared_conv_dir = "shared_conversations"
                    os.makedirs(shared_conv_dir, exist_ok=True)
                    file_path = os.path.join(shared_conv_dir, f"{share_id}.json")
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(share_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"✅ Shared conversation created: {share_id} ({len(request_obj.share_history)} messages)")
                    
                    # Generate share URL - use current path or default to /test
                    # Note: Frontend will override this with actual pathname
                    share_url = f"/test?share={share_id}"
                    
                    # Send success response
                    success_data = {
                        "type": "share_complete",
                        "success": True,
                        "share_id": share_id,
                        "share_url": share_url,
                        "expires_at": share_data["expires_at"],
                        "message": "Share link created successfully!"
                    }
                    yield f"data: {json.dumps(success_data, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Share conversation failed: {e}", exc_info=True)
                    error_event = json.dumps({
                        "type": "error",
                        "message": f"Share failed: {str(e)}"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
                    return
            
            # LOAD SHARED CONVERSATION MODE
            if request_obj.load_shared_conversation:
                logger.info(f"📥 Load shared conversation mode activated via /chat/stream")
                try:
                    if not request_obj.share_id:
                        raise ValueError("share_id is required when load_shared_conversation=true")
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Loading shared conversation...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Validate share ID format
                    if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", request_obj.share_id):
                        raise ValueError("Invalid share ID format")
                    
                    # Load from file
                    shared_conv_dir = "shared_conversations"
                    file_path = os.path.join(shared_conv_dir, f"{request_obj.share_id}.json")
                    
                    if not os.path.exists(file_path):
                        raise FileNotFoundError("Shared conversation not found or expired")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        share_data = json.load(f)
                    
                    # Check expiration
                    expires_at = share_data.get("expires_at")
                    if expires_at:
                        expires_timestamp = expires_at if isinstance(expires_at, int) else int(expires_at)
                        current_timestamp = int(datetime.now().timestamp() * 1000)
                        if current_timestamp > expires_timestamp:
                            os.remove(file_path)  # Clean up expired file
                            raise ValueError("Shared conversation has expired")
                    
                    conversation_data = share_data.get("conversation") or share_data
                    
                    # Extract conversation fields
                    conv_id = request_obj.share_id
                    conv_title = conversation_data.get("title", "Shared Conversation")
                    conv_timestamp = conversation_data.get("timestamp", conversation_data.get("created_at", 0))
                    conv_history = conversation_data.get("history", [])
                    conv_context = conversation_data.get("context")
                    
                    # Send conversation data in chunks to avoid SSE size limits
                    # First send metadata and history (smaller)
                    success_data = {
                        "type": "shared_conversation_loaded",
                        "success": True,
                        "conversation": {
                            "id": conv_id,
                            "title": conv_title,
                            "timestamp": conv_timestamp,
                            "history": conv_history,
                            "context": None  # Will be sent separately
                        },
                        "message": "Shared conversation loaded successfully!"
                    }
                    yield f"data: {json.dumps(success_data, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # If context exists, send it in chunks (similar to accumulated_context handling)
                    if conv_context:
                        # Serialize context to JSON string
                        if isinstance(conv_context, str):
                            ctx_string = conv_context
                        else:
                            ctx_string = json.dumps(conv_context, ensure_ascii=True)
                        
                        # Send context in chunks (max 4000 chars per chunk)
                        chunk_size = 4000
                        total_chunks = (len(ctx_string) + chunk_size - 1) // chunk_size
                        
                        for i in range(0, len(ctx_string), chunk_size):
                            chunk = ctx_string[i:i + chunk_size]
                            chunk_index = i // chunk_size
                            
                            chunk_data = {
                                "type": "shared_conversation_context_chunk",
                                "chunk_index": chunk_index,
                                "total_chunks": total_chunks,
                                "chunk": chunk
                            }
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=True)}\n\n"
                            await asyncio.sleep(0.01)
                        
                        # Send completion signal
                        complete_data = {
                            "type": "shared_conversation_context_complete",
                            "context": conv_context  # Send full context as object for convenience
                        }
                        yield f"data: {json.dumps(complete_data, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                    
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Load shared conversation failed: {e}", exc_info=True)
                    error_event = json.dumps({
                        "type": "error",
                        "message": f"Failed to load shared conversation: {str(e)}"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
                    return
            
            # SUBMIT FEEDBACK MODE
            if request_obj.submit_feedback:
                logger.info(f"📝 Submit feedback mode activated via /chat/stream")
                try:
                    # Validate feedback data
                    if not request_obj.conversation_id:
                        raise ValueError("conversation_id is required when submit_feedback=true")
                    if request_obj.message_index is None:
                        raise ValueError("message_index is required when submit_feedback=true")
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Saving feedback...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Create feedback data
                    feedback_entry = {
                        "conversation_id": request_obj.conversation_id,
                        "message_index": request_obj.message_index,
                        "vote": request_obj.vote,
                        "reported": request_obj.reported,
                        "timestamp": request_obj.timestamp or int(datetime.now().timestamp() * 1000),
                        "saved_at": int(datetime.now().timestamp() * 1000)
                    }
                    
                    # Store feedback in shared_conversations directory
                    # We'll create/update a feedback file for each conversation
                    feedback_dir = "shared_conversations/feedback"
                    os.makedirs(feedback_dir, exist_ok=True)
                    feedback_file = os.path.join(feedback_dir, f"{request_obj.conversation_id}.json")
                    
                    # Load existing feedback for this conversation or create new
                    if os.path.exists(feedback_file):
                        with open(feedback_file, 'r', encoding='utf-8') as f:
                            feedback_data = json.load(f)
                    else:
                        feedback_data = {
                            "conversation_id": request_obj.conversation_id,
                            "feedback": {}
                        }
                    
                    # Update feedback for this message
                    msg_id = f"msg-{request_obj.message_index}"
                    feedback_data["feedback"][msg_id] = feedback_entry
                    feedback_data["last_updated"] = int(datetime.now().timestamp() * 1000)
                    
                    # Save feedback file
                    with open(feedback_file, 'w', encoding='utf-8') as f:
                        json.dump(feedback_data, f, ensure_ascii=False, indent=2)
                    
                    vote_type = request_obj.vote if request_obj.vote else "reported" if request_obj.reported else "feedback"
                    logger.info(f"✅ Feedback saved: {request_obj.conversation_id} - message {request_obj.message_index} - {vote_type}")
                    
                    # Send success response
                    success_data = {
                        "type": "feedback_complete",
                        "success": True,
                        "message": "Feedback saved successfully!"
                    }
                    yield f"data: {json.dumps(success_data, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Submit feedback failed: {e}", exc_info=True)
                    error_event = json.dumps({
                        "type": "error",
                        "message": f"Failed to save feedback: {str(e)}"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
                    return
            
            # DELETE DOCUMENTS MODE
            if request_obj.delete_documents:
                logger.info(f"🗑️ Delete documents mode activated via /chat/stream")
                try:
                    # Validate deletion parameters
                    if not request_obj.delete_type:
                        raise ValueError("delete_type is required when delete_documents=true")
                    if not request_obj.delete_identifier:
                        raise ValueError("delete_identifier is required when delete_documents=true")
                    
                    delete_type = request_obj.delete_type
                    delete_identifier = request_obj.delete_identifier.strip()
                    
                    logger.info(f"🗑️ Deleting documents: type={delete_type}, identifier={delete_identifier}")
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Searching for documents to delete...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Build Qdrant filter based on document type
                    filter_conditions = []
                    
                    if delete_type == "news":
                        # News articles: type="news", delete by URL
                        filter_conditions = [
                            {"key": "type", "match": {"value": "news"}},
                            {"key": "url", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "doc":
                        # Doc documents: type="doc", delete by title
                        filter_conditions = [
                            {"key": "type", "match": {"value": "doc"}},
                            {"key": "title", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "custom_pdf":
                        # Custom PDF: type="custom", document_type="pdf", delete by title
                        filter_conditions = [
                            {"key": "type", "match": {"value": "custom"}},
                            {"key": "document_type", "match": {"value": "pdf"}},
                            {"key": "title", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "custom_instagram":
                        # Custom Instagram: type="custom", document_type="instagram", delete by URL
                        filter_conditions = [
                            {"key": "type", "match": {"value": "custom"}},
                            {"key": "document_type", "match": {"value": "instagram"}},
                            {"key": "url", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "custom_url":
                        # Custom URL: type="custom", document_type="url", delete by URL
                        filter_conditions = [
                            {"key": "type", "match": {"value": "custom"}},
                            {"key": "document_type", "match": {"value": "url"}},
                            {"key": "url", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "custom_youtube":
                        # Custom YouTube: type="custom", document_type="youtube"
                        # Some YouTube videos have URL stored, some only have title
                        # Use OR condition to match either URL or title
                        filter_conditions = [
                            {"key": "type", "match": {"value": "custom"}},
                            {"key": "document_type", "match": {"value": "youtube"}}
                        ]
                        # Add OR condition for URL or title match
                        # Note: Qdrant doesn't support OR at top level, so we need to use should
                        # But for simplicity, let's try URL first, then fall back to title
                        # Actually, we'll search for both and let the user know
                        
                        # First try to find by URL
                        url_filter = filter_conditions + [{"key": "url", "match": {"value": delete_identifier}}]
                        # Also prepare title filter as fallback
                        title_filter = filter_conditions + [{"key": "title", "match": {"value": delete_identifier}}]
                        
                        # We'll use url_filter first, and if no results, try title_filter
                        # Store both for later use
                        youtube_url_filter = url_filter
                        youtube_title_filter = title_filter
                        filter_conditions = url_filter  # Start with URL filter

                    elif delete_type == "custom_text":
                        # Custom text: type="custom", document_type="text", delete by title
                        filter_conditions = [
                            {"key": "type", "match": {"value": "custom"}},
                            {"key": "document_type", "match": {"value": "text"}},
                            {"key": "title", "match": {"value": delete_identifier}}
                        ]
                    elif delete_type == "youtube_transcript":
                        # YouTube transcript: type="youtube_transcript", delete by video_url
                        filter_conditions = [
                            {"key": "type", "match": {"value": "youtube_transcript"}},
                            {"key": "video_url", "match": {"value": delete_identifier}}
                        ]
                    else:
                        raise ValueError(f"Invalid delete_type: {delete_type}")

                    
                    # First, count how many points will be deleted (using scroll to count)
                    from config import Config as ConfigClass
                    qdrant_url = ConfigClass.QDRANT_URL
                    collection_name = ConfigClass.QDRANT_COLLECTION
                    
                    scroll_url = f"{qdrant_url}/collections/{collection_name}/points/scroll"
                    scroll_payload = {
                        "filter": {"must": filter_conditions},
                        "limit": 10000,  # Get all matching points
                        "with_payload": False,
                        "with_vector": False
                    }
                    
                    scroll_response = requests.post(scroll_url, json=scroll_payload, timeout=30)
                    if scroll_response.status_code != 200:
                        raise Exception(f"Failed to query Qdrant: {scroll_response.text}")
                    
                    scroll_result = scroll_response.json()
                    points_to_delete = scroll_result.get("result", {}).get("points", [])
                    count = len(points_to_delete)
                    
                    logger.info(f"🔍 Found {count} points to delete")
                    
                    # Special handling for YouTube: if no results with URL filter, try title filter
                    if count == 0 and delete_type == "custom_youtube" and 'youtube_title_filter' in locals():
                        logger.info(f"🔄 No results with URL filter, trying title filter for YouTube...")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'No results by URL, trying by title...'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        
                        # Try with title filter
                        scroll_payload["filter"]["must"] = youtube_title_filter
                        scroll_response = requests.post(scroll_url, json=scroll_payload, timeout=30)
                        
                        if scroll_response.status_code != 200:
                            raise Exception(f"Failed to query Qdrant with title filter: {scroll_response.text}")
                        
                        scroll_result = scroll_response.json()
                        points_to_delete = scroll_result.get("result", {}).get("points", [])
                        count = len(points_to_delete)
                        
                        logger.info(f"🔍 Found {count} points with title filter")
                        
                        # Update filter_conditions to use title filter for deletion
                        if count > 0:
                            filter_conditions = youtube_title_filter
                    
                    if count == 0:
                        yield f"data: {json.dumps({'type': 'delete_complete', 'success': True, 'deleted_count': 0, 'message': 'No matching documents found.'}, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0.01)
                        return
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Found {count} chunks to delete. Deleting...'}, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    # Delete points using Qdrant's filter-based delete API
                    delete_url = f"{qdrant_url}/collections/{collection_name}/points/delete"
                    delete_payload = {
                        "filter": {"must": filter_conditions}
                    }
                    
                    delete_response = requests.post(delete_url, json=delete_payload, timeout=60)
                    
                    if delete_response.status_code != 200:
                        raise Exception(f"Qdrant deletion failed: {delete_response.text}")
                    
                    logger.info(f"✅ Successfully deleted {count} points from Qdrant")
                    
                    # Send success response
                    success_data = {
                        "type": "delete_complete",
                        "success": True,
                        "deleted_count": count,
                        "message": f"Successfully deleted {count} chunks!"
                    }
                    yield f"data: {json.dumps(success_data, ensure_ascii=True)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Delete documents failed: {e}", exc_info=True)
                    error_event = json.dumps({
                        "type": "error",
                        "message": f"Deletion failed: {str(e)}"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
                    return

            
            # NORMAL CHAT MODE
            provider = request_obj.provider or "VLLM"
            logger.info(f"Streaming question: {request_obj.question} (provider: {provider})")
            
            # Parse accumulated_context if it's a string
            accumulated_context = request_obj.accumulated_context
            if accumulated_context and isinstance(accumulated_context, str):
                try:
                    accumulated_context = json.loads(accumulated_context)
                    logger.info("📦 Parsed accumulated_context from string to object")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse accumulated_context string: {e}")
                    accumulated_context = None
            
            # Debug: Log if accumulated_context was received
            if accumulated_context:
                total_sources = sum([
                    len(accumulated_context.get('vector_sources', [])),
                    len(accumulated_context.get('yacht_sources', [])),
                    len(accumulated_context.get('builder_sources', [])),
                    len(accumulated_context.get('country_sources', [])),
                    len(accumulated_context.get('global_sources', [])),
                    len(accumulated_context.get('model_sources', [])),
                    len(accumulated_context.get('yacht_detail_sources', [])),
                    len(accumulated_context.get('model_detail_sources', []))
                ])
                logger.info(f"📦 API received accumulated_context with {total_sources} sources from frontend")
            else:
                logger.info("📦 API received NO accumulated_context from frontend")
            
            # Convert conversation history to the format expected by the chatbot
            conversation_history = []
            if request_obj.conversation_history:
                for msg in request_obj.conversation_history:
                    msg_dict = {
                        'role': msg.role,
                        'content': msg.content if msg.content is not None else None
                    }
                    # Include tool_calls if present (for assistant messages with tool calls)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls is not None:
                        msg_dict['tool_calls'] = msg.tool_calls
                    # Include name if present (for tool messages)
                    if hasattr(msg, 'name') and msg.name is not None:
                        msg_dict['name'] = msg.name
                    conversation_history.append(msg_dict)
            
            # Create streaming chatbot instance with selected provider
            active_streaming_chatbot = StreamingRAGChatbot(provider=provider, enable_exa_search=request_obj.enable_exa_search)
            
            # Set custom system prompt if provided
            if request_obj.custom_system_prompt:
                active_streaming_chatbot.custom_system_prompt = request_obj.custom_system_prompt
                logger.info(f"✨ Using custom system prompt (length: {len(request_obj.custom_system_prompt)}): {request_obj.custom_system_prompt[:200]}...")
            else:
                logger.info("ℹ️ No custom system prompt provided")
            
            for event in active_streaming_chatbot.ask_stream(
                request_obj.question, 
                conversation_history,
                accumulated_context=accumulated_context
            ):
                try:
                    # Special handling for 'complete' event with accumulated_context
                    # Send accumulated_context in CHUNKS to avoid SSE size limits
                    if event.get('type') == 'complete' and event.get('data', {}).get('accumulated_context'):
                        # Extract and remove accumulated_context from complete event
                        accumulated_ctx = event['data']['accumulated_context']
                        del event['data']['accumulated_context']
                        
                        # Serialize to JSON string then base64 encode to avoid splitting issues
                        import base64
                        ctx_string = json.dumps(accumulated_ctx, ensure_ascii=True)
                        ctx_bytes = ctx_string.encode('utf-8')
                        ctx_base64 = base64.b64encode(ctx_bytes).decode('ascii')
                        
                        # Send in chunks (max 4000 chars per chunk to be safe)
                        # Base64 encoding ensures we won't split in the middle of escape sequences
                        chunk_size = 4000
                        total_chunks = (len(ctx_base64) + chunk_size - 1) // chunk_size
                        
                        for i in range(0, len(ctx_base64), chunk_size):
                            chunk = ctx_base64[i:i + chunk_size]
                            chunk_index = i // chunk_size
                            
                            ctx_chunk_event = {
                                "type": "accumulated_context_chunk",
                                "chunk_index": chunk_index,
                                "total_chunks": total_chunks,
                                "data": chunk,
                                "encoding": "base64"  # Tell frontend this is base64 encoded
                            }
                            chunk_event_data = json.dumps(ctx_chunk_event, ensure_ascii=True)
                            yield f"data: {chunk_event_data}\n\n"
                            await asyncio.sleep(0.01)
                    
                    # Format as Server-Sent Event with proper escaping
                    # Use ensure_ascii=True to properly escape all special characters
                    event_data = json.dumps(event, ensure_ascii=True)
                    # Send the event immediately - no delay for answer_chunk to ensure real-time streaming
                    yield f"data: {event_data}\n\n"
                    # Only add delay for non-answer_chunk events to prevent overwhelming the client
                    if event.get('type') != 'answer_chunk':
                        await asyncio.sleep(0.01)  # Small delay for other events (10ms)
                except Exception as json_error:
                    # Log the problematic event for debugging
                    logger.error(f"Error serializing event to JSON: {json_error}")
                    logger.error(f"Problematic event type: {event.get('type', 'unknown')}")
                    # Send a sanitized error event instead
                    error_event = json.dumps({
                        "type": "error",
                        "message": "Error serializing response data"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
            
            logger.info("Streaming completed successfully")
            
        except BrokenPipeError:
            # Client disconnected - this is expected, just log and exit gracefully
            logger.info("[INFO] Client disconnected during API streaming (BrokenPipeError) - this is normal")
            return
        except Exception as e:
            # Only show error to user if it's not a broken pipe error
            logger.error(f"Error during streaming: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't try to send error if client disconnected
            try:
                error_event = json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                }, ensure_ascii=True)
                yield f"data: {error_event}\n\n"
            except (BrokenPipeError, ConnectionError):
                # Client already disconnected, can't send error
                logger.info("[INFO] Could not send error to client - connection already closed")
                return
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )

@app.post("/chat-url", response_model=ChatResponse)
async def chat_url(request: ChatRequest):
    """
    Chat with the URL creation chatbot (uses prompts from prompts_url_creation.py)
    
    This endpoint processes a user's request to generate URLs and returns a comprehensive answer
    based on multiple data sources. It uses specialized prompts for URL generation.
    """
    try:
        provider = request.provider or "VLLM"
        logger.info(f"Processing URL creation question: {request.question} (provider: {provider})")
        
        # Parse accumulated_context if it's a string
        accumulated_context = request.accumulated_context
        if accumulated_context and isinstance(accumulated_context, str):
            try:
                accumulated_context = json.loads(accumulated_context)
                logger.info("📦 Parsed accumulated_context from string to object")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse accumulated_context string: {e}")
                accumulated_context = None
        
        # Convert conversation history to the format expected by the chatbot
        conversation_history = []
        if request.conversation_history:
            for msg in request.conversation_history:
                conversation_history.append({
                    'role': msg.role,
                    'content': msg.content
                })
        
        # Create chatbot instance with URL creation prompts (clarification disabled)
        active_chatbot = SimpleRAGChatbot(provider=provider, prompts_module=prompts_url_creation, enable_clarification=False, enable_exa_search=request.enable_exa_search)
        
        # Get response from chatbot with accumulated context
        response = active_chatbot.ask(
            request.question, 
            conversation_history,
            accumulated_context=accumulated_context
        )
        
        # Build updated conversation history
        updated_history = []
        if request.conversation_history:
            updated_history.extend([ChatMessage(role=msg.role, content=msg.content) for msg in request.conversation_history])
        
        # Add current question and answer
        updated_history.append(ChatMessage(role="user", content=request.question))
        updated_history.append(ChatMessage(role="assistant", content=response["answer"]))
        
        # Strip "URL: " prefix from the answer if present (for wrapper API compatibility)
        answer = response["answer"]
        if answer.startswith("URL: "):
            answer = answer[5:].strip()
            logger.info(f"🔗 Stripped 'URL: ' prefix from answer")
        
        # Convert to API response format
        api_response = ChatResponse(
            question=response["question"],
            answer=answer,
            graphs=response.get("graphs", []),
            sources=SourceInfo(**response["sources"]),
            sources_display=response.get("sources_display", []),
            total_sources=response["total_sources"],
            context_length=response["context_length"],
            conversation_history=updated_history,
            accumulated_context=response.get("accumulated_context", None)
        )
        
        logger.info(f"Successfully processed URL creation question with {api_response.total_sources} sources")
        logger.info(f"📤 FINAL API RESPONSE (URL) - Answer: {api_response.answer}")
        logger.info(f"📤 FINAL API RESPONSE (URL) - Total sources: {api_response.total_sources}, Context length: {api_response.context_length}")
        return api_response
        
    except Exception as e:
        logger.error(f"Error processing chat-url request: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/chat-url/stream")
async def chat_url_stream(request: ChatRequest):
    """
    Stream chat responses for URL creation with real-time progress updates (uses prompts from prompts_url_creation.py)
    
    This endpoint uses Server-Sent Events (SSE) to stream URL creation responses with specialized prompts.
    """
    async def event_generator():
        """Generate Server-Sent Events with immediate flushing"""
        import asyncio
        try:
            provider = request.provider or "VLLM"
            logger.info(f"Streaming URL creation question: {request.question} (provider: {provider})")
            
            # Parse accumulated_context if it's a string
            accumulated_context = request.accumulated_context
            if accumulated_context and isinstance(accumulated_context, str):
                try:
                    accumulated_context = json.loads(accumulated_context)
                    logger.info("📦 Parsed accumulated_context from string to object")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse accumulated_context string: {e}")
                    accumulated_context = None
            
            # Debug: Log if accumulated_context was received
            if accumulated_context:
                total_sources = sum([
                    len(accumulated_context.get('vector_sources', [])),
                    len(accumulated_context.get('yacht_sources', [])),
                    len(accumulated_context.get('builder_sources', [])),
                    len(accumulated_context.get('country_sources', [])),
                    len(accumulated_context.get('global_sources', [])),
                    len(accumulated_context.get('model_sources', [])),
                    len(accumulated_context.get('yacht_detail_sources', [])),
                    len(accumulated_context.get('model_detail_sources', []))
                ])
                logger.info(f"📦 API received accumulated_context with {total_sources} sources from frontend")
            else:
                logger.info("📦 API received NO accumulated_context from frontend")
            
            # Convert conversation history to the format expected by the chatbot
            conversation_history = []
            if request.conversation_history:
                for msg in request.conversation_history:
                    conversation_history.append({
                        'role': msg.role,
                        'content': msg.content
                    })
            
            # Create streaming chatbot instance with URL creation prompts (clarification disabled)
            active_streaming_chatbot = StreamingRAGChatbot(provider=provider, prompts_module=prompts_url_creation, enable_clarification=False, enable_exa_search=request.enable_exa_search)
            
            # Track if we've seen "URL: " prefix to strip it from streaming chunks
            url_prefix_stripped = False
            
            for event in active_streaming_chatbot.ask_stream(
                request.question, 
                conversation_history,
                accumulated_context=accumulated_context
            ):
                try:
                    # Strip "URL: " prefix from answer chunks/complete events (for wrapper API compatibility)
                    if event.get('type') == 'answer_chunk' and not url_prefix_stripped:
                        chunk = event.get('chunk', '')
                        if chunk.startswith("URL: "):
                            event['chunk'] = chunk[5:].strip()
                            url_prefix_stripped = True
                            logger.info(f"🔗 Stripped 'URL: ' prefix from streaming answer")
                    elif event.get('type') == 'answer_complete':
                        answer = event.get('answer', '')
                        if answer.startswith("URL: "):
                            event['answer'] = answer[5:].strip()
                            logger.info(f"🔗 Stripped 'URL: ' prefix from complete answer")
                    elif event.get('type') == 'complete' and event.get('data', {}).get('answer'):
                        answer = event['data'].get('answer', '')
                        if answer.startswith("URL: "):
                            event['data']['answer'] = answer[5:].strip()
                    
                    # Special handling for 'complete' event with accumulated_context
                    # Send accumulated_context in CHUNKS to avoid SSE size limits
                    if event.get('type') == 'complete' and event.get('data', {}).get('accumulated_context'):
                        # Extract and remove accumulated_context from complete event
                        accumulated_ctx = event['data']['accumulated_context']
                        del event['data']['accumulated_context']
                        
                        # Serialize to JSON string
                        ctx_string = json.dumps(accumulated_ctx, ensure_ascii=True)
                        
                        # Send in chunks (max 4000 chars per chunk to be safe)
                        chunk_size = 4000
                        total_chunks = (len(ctx_string) + chunk_size - 1) // chunk_size
                        
                        for i in range(0, len(ctx_string), chunk_size):
                            chunk = ctx_string[i:i + chunk_size]
                            chunk_index = i // chunk_size
                            
                            ctx_chunk_event = {
                                "type": "accumulated_context_chunk",
                                "chunk_index": chunk_index,
                                "total_chunks": total_chunks,
                                "data": chunk
                            }
                            chunk_event_data = json.dumps(ctx_chunk_event, ensure_ascii=True)
                            yield f"data: {chunk_event_data}\n\n"
                            await asyncio.sleep(0.01)
                    
                    # Format as Server-Sent Event with proper escaping
                    # Use ensure_ascii=True to properly escape all special characters
                    event_data = json.dumps(event, ensure_ascii=True)
                    # Send the event immediately - no delay for answer_chunk to ensure real-time streaming
                    yield f"data: {event_data}\n\n"
                    # Only add delay for non-answer_chunk events to prevent overwhelming the client
                    if event.get('type') != 'answer_chunk':
                        await asyncio.sleep(0.01)  # Small delay for other events (10ms)
                except Exception as json_error:
                    # Log the problematic event for debugging
                    logger.error(f"Error serializing event to JSON: {json_error}")
                    logger.error(f"Problematic event type: {event.get('type', 'unknown')}")
                    # Send a sanitized error event instead
                    error_event = json.dumps({
                        "type": "error",
                        "message": "Error serializing response data"
                    }, ensure_ascii=True)
                    yield f"data: {error_event}\n\n"
            
            logger.info("URL creation streaming completed successfully")
            
        except BrokenPipeError:
            # Client disconnected - this is expected, just log and exit gracefully
            logger.info("[INFO] Client disconnected during API streaming (BrokenPipeError) - this is normal")
            return
        except Exception as e:
            # Only show error to user if it's not a broken pipe error
            logger.error(f"Error during URL creation streaming: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't try to send error if client disconnected
            try:
                error_event = json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                }, ensure_ascii=True)
                yield f"data: {error_event}\n\n"
            except (BrokenPipeError, ConnectionError):
                # Client already disconnected, can't send error
                logger.info("[INFO] Could not send error to client - connection already closed")
                return
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )

@app.get("/test", response_class=HTMLResponse)
async def serve_test_page():
    """Serve the streaming test HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "test_streaming.html")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")

@app.get("/test-url", response_class=HTMLResponse)
async def serve_test_url_page():
    """Serve the URL creation streaming test HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "test_streaming_url.html")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="URL test page not found")

@app.get("/simple", response_class=HTMLResponse)
async def serve_simple_test_page():
    """Serve the simple streaming test HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "simple_stream_test.html")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Simple test page not found")

@app.post("/chunk-semantic-llm", response_model=SemanticChunkResponse)
async def chunk_semantic_llm(request: SemanticChunkRequest):
    """
    Chunk text semantically using LLM (Qwen model).
    The LLM analyzes the text and creates semantically meaningful chunks
    that are suitable for vector database storage.
    """
    try:
        from services.llm_service import LLMService
        
        # Create LLM service instance (will use VLLM/Qwen by default)
        llm_service = LLMService(provider="VLLM")
        
        # Create the semantic chunking prompt
        chunking_prompt = f"""You are an expert semantic text chunker. Your task is to split the provided text into semantically meaningful chunks that are suitable for vector database storage.

CRITICAL INSTRUCTIONS:
1. Analyze the text and identify natural semantic boundaries (topics, sections, paragraphs, concepts)
2. Create chunks that are semantically coherent - each chunk should contain related information
3. Each chunk should be between {request.min_chunk_size} and {request.max_chunk_size} characters (these are guidelines, prioritize semantic coherence)
4. Do NOT split sentences or paragraphs in the middle - keep complete thoughts together
5. Each chunk should be self-contained and meaningful on its own
6. Preserve the original text exactly - do not modify, summarize, or paraphrase

OUTPUT FORMAT (CRITICAL - YOU MUST FOLLOW THIS EXACT FORMAT):
You must output ONLY valid JSON in this exact structure:
```json
{{
  "chunks": [
    "First chunk of text here...",
    "Second chunk of text here...",
    "Third chunk of text here..."
  ],
  "metadata": {{
    "total_characters": 1234,
    "avg_chunk_size": 400,
    "reasoning": "Brief explanation of chunking strategy used"
  }}
}}
```

IMPORTANT:
- Output ONLY the JSON, no additional text before or after
- Each chunk in the "chunks" array should be a complete string
- The chunks array must contain at least 1 chunk
- All chunks combined should cover the entire input text (no text should be lost)

TEXT TO CHUNK:
{request.text}

Now analyze this text and create semantically meaningful chunks. Output ONLY the JSON response:"""

        logger.info(f"🧠 Calling LLM for semantic chunking (text length: {len(request.text)} chars)")
        
        # Call the LLM without tools (we want direct text response)
        response = llm_service.call_llm(
            user_message=chunking_prompt,
            use_tools=False,
            stream=False,
            no_think=True,
            use_research_model=False
        )
        
        # Extract the content from the response
        if "choices" not in response or len(response["choices"]) == 0:
            raise HTTPException(status_code=500, detail="LLM returned invalid response format")
        
        content = response["choices"][0]["message"].get("content", "")
        
        if not content:
            raise HTTPException(status_code=500, detail="LLM returned empty response")
        
        logger.info(f"📄 LLM response received (length: {len(content)} chars)")
        logger.debug(f"📄 LLM response preview: {content[:500]}...")
        
        # Parse the JSON response from the LLM
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content.strip()
        
        # Parse the JSON
        try:
            parsed_response = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM JSON response: {e}")
            logger.error(f"📄 Raw content: {content[:1000]}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse LLM response as JSON: {str(e)}. LLM may not have followed the format."
            )
        
        # Validate the response structure
        if "chunks" not in parsed_response or not isinstance(parsed_response["chunks"], list):
            raise HTTPException(
                status_code=500,
                detail="LLM response missing 'chunks' array. Response may not have followed the format."
            )
        
        chunks = parsed_response["chunks"]
        if len(chunks) == 0:
            raise HTTPException(
                status_code=500,
                detail="LLM returned empty chunks array"
            )
        
        # Validate chunks are strings
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise HTTPException(
                    status_code=500,
                    detail=f"Chunk {i} is not a string (got {type(chunk).__name__})"
                )
        
        metadata = parsed_response.get("metadata", {})
        metadata["llm_provider"] = "VLLM"
        metadata["llm_model"] = llm_service._get_model(use_research_model=False)
        
        logger.info(f"✅ Successfully created {len(chunks)} semantic chunks")
        
        return SemanticChunkResponse(
            chunks=chunks,
            chunk_count=len(chunks),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in semantic chunking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic chunking failed: {str(e)}")

@app.post("/transcribe-audio", response_model=AudioTranscriptionResponse)
async def transcribe_audio(request: AudioTranscriptionRequest):
    """
    Transcribe audio using Gemini model with optional guidance prompt.
    The system prompt instructs the model to transcribe accurately without commentary.
    The guidance prompt allows users to specify terms, context, or special instructions.
    """
    try:
        from services.llm_service import LLMService
        import base64
        
        # Create LLM service instance with Gemini provider
        llm_service = LLMService(provider="gemini")
        
        logger.info(f"🎤 Starting audio transcription (format: {request.audio_format}, guidance: {request.guidance_prompt is not None})")
        
        # Call the transcription method
        response = llm_service.transcribe_audio(
            audio_base64=request.audio_base64,
            audio_format=request.audio_format,
            guidance_prompt=request.guidance_prompt
        )
        
        # Check for errors
        if "error" in response:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {response['error']}")
        
        # Extract transcription from response
        if "choices" not in response or len(response["choices"]) == 0:
            raise HTTPException(status_code=500, detail="LLM returned invalid response format")
        
        transcription = response["choices"][0]["message"].get("content", "")
        
        if not transcription:
            raise HTTPException(status_code=500, detail="LLM returned empty transcription")
        
        logger.info(f"✅ Transcription completed (length: {len(transcription)} chars)")
        
        # Build metadata
        metadata = {
            "audio_format": request.audio_format,
            "audio_data_length": len(request.audio_base64),
            "guidance_used": request.guidance_prompt is not None,
            "llm_provider": "gemini",
            "llm_model": llm_service._get_model(use_research_model=False)
        }
        
        return AudioTranscriptionResponse(
            transcription=transcription,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Error in audio transcription: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error in audio transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

@app.post("/transcribe-youtube", response_model=YouTubeTranscriptionResponse)
async def transcribe_youtube(request: YouTubeTranscriptionRequest):
    """
    Transcribe YouTube video using youtube-transcript-api library.
    The endpoint takes a YouTube URL, fetches the transcript directly from YouTube,
    and returns a clean transcript without timestamps.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked
        
        # Validate YouTube URL
        youtube_url = request.youtube_url.strip()
        if not youtube_url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
        
        logger.info(f"🎬 Starting YouTube transcription for URL: {youtube_url}")
        
        # Extract video ID from URL
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if not video_id_match:
            raise HTTPException(status_code=400, detail="Could not extract video ID from YouTube URL")
        
        video_id = video_id_match.group(1)
        logger.info(f"📹 Extracted video ID: {video_id}")
        
        # Create API instance with proxy to bypass IP blocking
        from youtube_transcript_api.proxies import WebshareProxyConfig
        api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username="amxukcgo",
                proxy_password="q0nxvwzetxjz",
            )
        )
        
        transcript_text_parts = None
        try:
            # Fetch transcript (tries English first)
            fetched_transcript = api.fetch(video_id, languages=['en'])
            # Extract text from FetchedTranscriptSnippet objects
            transcript_text_parts = [snippet.text for snippet in fetched_transcript]
            logger.info(f"✅ Found English transcript with {len(transcript_text_parts)} entries")
        except NoTranscriptFound:
            try:
                # Try any available language
                logger.info("⚠️ No English transcript, trying other languages...")
                available_transcripts = api.list(video_id)
                for transcript in available_transcripts:
                    try:
                        fetched_transcript = transcript.fetch()
                        transcript_text_parts = [snippet.text for snippet in fetched_transcript]
                        logger.info(f"✅ Found transcript in language: {transcript.language}")
                        break
                    except:
                        continue
                
                if not transcript_text_parts:
                    raise NoTranscriptFound("No transcript found in any language")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"No transcript available for this video: {str(e)}")
        except TranscriptsDisabled:
            raise HTTPException(status_code=400, detail="Transcripts are disabled for this video")
        except VideoUnavailable:
            raise HTTPException(status_code=404, detail="Video is unavailable or private")
        except IpBlocked:
            logger.warning(f"⚠️ YouTube IP blocked, falling back to Gemini...")
            # Fallback to Gemini when IP is blocked
            from config import Config
            config = Config()
            
            model_id = "gemini-3-flash-preview"
            api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
            
            user_prompt = f"""Please fetch and transcribe the audio from this YouTube video: {youtube_url}

Requirements:
- Use Google Search tool to access the video
- Extract the full transcript
- Return ONLY the clean text transcript
- Do NOT include timestamps
- Do NOT include any commentary or additional text
- Just the pure transcript text"""

            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}},
                "tools": [{"googleSearch": {}}]
            }
            
            url = f"{api_endpoint}?key={config.GEMINI_API_KEY}"
            transcription_response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=600)
            
            if transcription_response.status_code != 200:
                raise HTTPException(status_code=503, detail=f"YouTube API blocked and Gemini fallback failed: {transcription_response.status_code}")
            
            response_json = transcription_response.json()
            transcription = ""
            
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate:
                    content = candidate["content"]
                    if "parts" in content:
                        for part in content["parts"]:
                            if "text" in part:
                                transcription = part["text"].strip()
                                break
                    elif "text" in content:
                        transcription = content["text"].strip()
            
            if not transcription or len(transcription) < 100:
                raise HTTPException(status_code=503, detail="Both YouTube API and Gemini fallback failed")
            
            # Check for error messages
            error_indicators = ["I am unable to", "I cannot", "cannot fetch", "unable to fetch"]
            if any(indicator in transcription.lower() for indicator in error_indicators):
                raise HTTPException(status_code=503, detail=f"Both methods failed. Gemini returned: {transcription[:200]}")
            
            logger.info(f"✅ Gemini fallback transcription complete: {len(transcription)} chars")
        except Exception as e:
            logger.error(f"❌ Error fetching transcript: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")
        
        # Combine transcript text parts into clean text
        transcription = " ".join(transcript_text_parts)
        
        # Clean up common transcript artifacts
        transcription = transcription.replace('\n', ' ')
        transcription = re.sub(r'\s+', ' ', transcription)  # Remove extra whitespace
        transcription = transcription.strip()
        
        logger.info(f"✅ YouTube transcription completed (length: {len(transcription)} chars)")
        logger.info(f"📝 TRANSCRIPT PREVIEW (first 500 chars):\n{transcription[:500]}")
        
        # Build metadata
        metadata = {
            "youtube_url": youtube_url,
            "video_id": video_id,
            "transcript_provider": "youtube-transcript-api",
            "transcript_entries": len(transcript_text_parts)
        }
        
        return YouTubeTranscriptionResponse(
            transcription=transcription,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Error in YouTube transcription: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error in YouTube transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YouTube transcription failed: {str(e)}")

class YouTubeChannelScraperRequest(BaseModel):
    channel_url: str = Field(..., description="YouTube channel URL to scrape (e.g., https://www.youtube.com/@Yachtbuyer)")
    max_videos: int = Field(default=10, description="Maximum number of regular videos to scrape")
    max_shorts: int = Field(default=0, description="Maximum number of shorts to scrape")
    max_streams: int = Field(default=0, description="Maximum number of streams to scrape")
    save_to_file: Optional[bool] = Field(default=False, description="If true, save results to a text file")
    
    class Config:
        schema_extra = {
            "example": {
                "channel_url": "https://www.youtube.com/@Yachtbuyer",
                "max_videos": 10,
                "max_shorts": 0,
                "max_streams": 0,
                "save_to_file": True
            }
        }

class YouTubeChannelScraperResponse(BaseModel):
    success: bool = Field(description="Whether the scraping was successful")
    channel_info: Optional[Dict[str, Any]] = Field(default=None, description="Channel information")
    videos: List[Dict[str, Any]] = Field(default=[], description="List of video information")
    total_videos: int = Field(description="Total number of videos scraped")
    file_path: Optional[str] = Field(default=None, description="Path to saved file if save_to_file was true")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

@app.post("/scrape-youtube-channel", response_model=YouTubeChannelScraperResponse)
async def scrape_youtube_channel(request: YouTubeChannelScraperRequest):
    """
    Scrape YouTube channel to get all video URLs and information.
    
    This endpoint uses Apify's YouTube Channel Scraper to fetch:
    - Channel information (name, subscribers, description, etc.)
    - Video URLs, titles, thumbnails, view counts
    - Video metadata (duration, date, etc.)
    
    Optionally saves results to a text file with all URLs.
    """
    try:
        from apify_client import ApifyClient
        from config import Config as ConfigClass
        
        logger.info(f"🎬 Starting YouTube channel scraper for: {request.channel_url}")
        
        # Get API token from config
        apify_token = ConfigClass.APIFY_API_TOKEN
        if not apify_token:
            raise ValueError("APIFY_API_TOKEN not configured. Please set it in config.py or environment variable.")
        
        # Initialize Apify client
        client = ApifyClient(apify_token)
        
        # Prepare Actor input
        run_input = {
            "startUrls": [
                {
                    "url": request.channel_url
                }
            ],
            "maxResults": request.max_videos,
            "maxResultsShorts": request.max_shorts,
            "maxResultStreams": request.max_streams
        }
        
        logger.info(f"🚀 Starting Apify Actor for YouTube Channel Scraper...")
        logger.info(f"📊 Parameters: max_videos={request.max_videos}, max_shorts={request.max_shorts}, max_streams={request.max_streams}")
        
        # Run the Actor and wait for it to finish
        run = client.actor("streamers/youtube-channel-scraper").call(run_input=run_input)
        
        logger.info(f"✅ Apify Actor completed. Fetching results...")
        
        # Fetch results from the dataset
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if not items:
            raise ValueError("No results returned from Apify Actor")
        
        logger.info(f"✅ Found {len(items)} items from channel")
        
        # Extract channel info from the first item (aboutChannelInfo field)
        channel_info = None
        if items and 'aboutChannelInfo' in items[0]:
            channel_info = items[0]['aboutChannelInfo']
            logger.info(f"📺 Channel: {channel_info.get('channelName', 'Unknown')}")
            logger.info(f"👥 Subscribers: {channel_info.get('numberOfSubscribers', 'Unknown')}")
        
        # Process video items
        videos = []
        for item in items:
            video_data = {
                'id': item.get('id'),
                'title': item.get('title'),
                'url': item.get('url'),
                'duration': item.get('duration'),
                'date': item.get('date'),
                'viewCount': item.get('viewCount'),
                'thumbnailUrl': item.get('thumbnailUrl'),
                'type': item.get('type')  # video, short, or stream
            }
            videos.append(video_data)
        
        logger.info(f"✅ Processed {len(videos)} videos")
        
        # Save to file if requested
        file_path = None
        if request.save_to_file:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"youtube_channel_urls_{timestamp}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"YouTube Channel: {request.channel_url}\n")
                f.write(f"Scraped at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                if channel_info:
                    f.write("CHANNEL INFORMATION:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Name: {channel_info.get('channelName', 'N/A')}\n")
                    f.write(f"Subscribers: {channel_info.get('numberOfSubscribers', 'N/A')}\n")
                    f.write(f"Total Videos: {channel_info.get('channelTotalVideos', 'N/A')}\n")
                    f.write(f"Total Views: {channel_info.get('channelTotalViews', 'N/A')}\n")
                    f.write(f"Description: {channel_info.get('channelDescription', 'N/A')}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
                
                f.write(f"VIDEO URLS ({len(videos)} videos):\n")
                f.write("-" * 80 + "\n\n")
                
                for idx, video in enumerate(videos, 1):
                    f.write(f"{idx}. {video.get('title', 'Untitled')}\n")
                    f.write(f"   URL: {video.get('url', 'N/A')}\n")
                    f.write(f"   Type: {video.get('type', 'N/A')}\n")
                    f.write(f"   Duration: {video.get('duration', 'N/A')}\n")
                    f.write(f"   Views: {video.get('viewCount', 'N/A')}\n")
                    f.write(f"   Date: {video.get('date', 'N/A')}\n")
                    f.write("\n")
            
            logger.info(f"💾 Results saved to: {file_path}")
        
        return YouTubeChannelScraperResponse(
            success=True,
            channel_info=channel_info,
            videos=videos,
            total_videos=len(videos),
            file_path=file_path,
            metadata={
                "run_id": run.get("id"),
                "dataset_id": run.get("defaultDatasetId")
            }
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Error in YouTube channel scraper: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error in YouTube channel scraper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YouTube channel scraping failed: {str(e)}")

@app.get("/chat/example", response_model=Dict[str, Any])
async def get_example_questions():
    """Get example questions to try with the chatbot"""
    return {
        "examples": [
            "What are the specifications of Princess V50?",
            "Find me a 50-meter yacht for sale",
            "What engines does the Azimut Grande 32M have?",
            "Tell me about Ferretti Group",
            "What's the difference between a motor yacht and a sailing yacht?",
            "Show me luxury yachts built by Benetti",
            "What's the price range for a 40-meter yacht?",
            "Find yachts with helicopter landing pads"
        ]
    }

# Custom Document Upload Models and Endpoints
class CustomDocumentUploadRequest(BaseModel):
    content: str = Field(..., description="The document content - either base64 encoded PDF or plain text")
    content_type: str = Field(..., description="Content type: 'pdf' or 'text'")
    title: Optional[str] = Field(None, description="Optional title for the document")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "This is a plain text document about yachts...",
                "content_type": "text",
                "title": "Yacht Industry Report 2024"
            }
        }

class CustomDocumentUploadResponse(BaseModel):
    success: bool = Field(description="Whether the upload was successful")
    message: str = Field(description="Status message")
    chunks_created: int = Field(description="Number of chunks created")
    points_inserted: int = Field(description="Number of points inserted into Qdrant")
    document_id: str = Field(description="Unique identifier for this document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

@app.post("/upload-custom-document", response_model=CustomDocumentUploadResponse)
async def upload_custom_document(request: CustomDocumentUploadRequest):
    """
    Upload a custom document (PDF or plain text) to the knowledge base.
    
    Process:
    1. If PDF: Convert to images and OCR each page
    2. If text: Use directly
    3. Chunk text using word-count algorithm (default: 200 words/chunk, 30 overlap)
    4. Generate embeddings for each chunk
    5. Store in Qdrant with type="custom" and optional title
    """
    import base64
    import fitz  # PyMuPDF
    import uuid
    
    try:
        logger.info(f"📥 Custom document upload started: type={request.content_type}, title={request.title}")
        
        # Step 1: Extract text from content
        extracted_text = ""
        
        if request.content_type == "pdf":
            logger.info("📄 Processing PDF document...")
            try:
                # Decode base64 PDF
                pdf_bytes = base64.b64decode(request.content)
                
                # Open PDF with PyMuPDF
                pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = len(pdf)
                logger.info(f"📄 PDF has {total_pages} pages")
                
                # OCR each page - use LLM URL from config
                from config import Config as ConfigClass
                ocr_url = ConfigClass.LLM_URL
                page_texts = []
                
                # Check if OCR service is available before processing
                # We'll catch connection errors on the first request and provide a clear error message
                logger.info(f"🔍 Checking OCR service availability at {ocr_url}...")
                
                for page_num in range(total_pages):
                    logger.info(f"🔍 OCR processing page {page_num + 1}/{total_pages}...")
                    page = pdf[page_num]
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Call OCR API
                    ocr_payload = {
                        "model": ConfigClass.LLM_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                                {"type": "text", "text": "Extract all text from this document image. Provide the complete text content exactly as it appears."}
                            ]
                        }],
                        "max_tokens": 4000
                    }
                    
                    try:
                        ocr_response = requests.post(ocr_url, json=ocr_payload, timeout=120)
                        if ocr_response.status_code != 200:
                            raise HTTPException(
                            status_code=500,
                            detail=f"OCR failed for page {page_num + 1}: {ocr_response.text}"
                            )
                    except requests.exceptions.ConnectionError as e:
                        raise HTTPException(
                            status_code=503,
                            detail=f"Failed to connect to OCR service at {ocr_url}. Please ensure the service is running. Error: {str(e)}"
                        )
                    except requests.exceptions.Timeout as e:
                        raise HTTPException(
                            status_code=504,
                            detail=f"OCR request timed out for page {page_num + 1}. The PDF might be too large or the service is overloaded. Error: {str(e)}"
                        )
                    except requests.exceptions.RequestException as e:
                        raise HTTPException(
                            status_code=500,
                            detail=f"OCR request failed for page {page_num + 1}: {str(e)}"
                        )
                    
                    ocr_result = ocr_response.json()
                    page_text = ocr_result['choices'][0]['message']['content']
                    page_texts.append(page_text)
                    logger.info(f"✅ Page {page_num + 1} OCR complete: {len(page_text)} chars")
                
                pdf.close()
                extracted_text = "\n\n".join(page_texts)
                logger.info(f"✅ PDF OCR complete: {len(extracted_text)} total characters")
                
            except Exception as e:
                logger.error(f"❌ PDF processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
        
        elif request.content_type == "text":
            logger.info("📝 Processing plain text document...")
            extracted_text = request.content
            logger.info(f"✅ Text loaded: {len(extracted_text)} characters")
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content_type: {request.content_type}. Must be 'pdf' or 'text'"
            )
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Extracted text is empty")
        
        # Step 2: Chunk text using word-count algorithm
        logger.info("✂️  Chunking text with word-count algorithm...")
        chunk_url = "http://localhost:8002/chunk"
        chunk_payload = {
            "text": extracted_text,
            "algorithm": "word-count"
            # Using default parameters: words_per_chunk=200, overlap_words=30
        }
        
        chunk_response = requests.post(chunk_url, json=chunk_payload, timeout=120)
        if chunk_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Chunking failed: {chunk_response.text}")
        
        chunk_result = chunk_response.json()
        chunks = chunk_result['chunks']
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        logger.info("🧮 Generating embeddings...")
        embed_url = "http://localhost:8080/embed"
        embed_payload = {"texts": chunks}
        
        embed_response = requests.post(embed_url, json=embed_payload, timeout=120)
        if embed_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {embed_response.text}")
        
        embed_result = embed_response.json()
        embeddings = embed_result['embeddings']
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        # Step 3.5: Generate sparse vectors for hybrid search
        logger.info("🔤 Generating sparse vectors...")
        # Use global chatbot instance to access sparse service
        if chatbot is None:
            logger.warning("⚠️ Chatbot not initialized, creating temporary instance for sparse vectors...")
            from agent import SimpleRAGChatbot
            temp_chatbot = SimpleRAGChatbot()
            sparse_service = temp_chatbot.vector_search_service.sparse_service
        else:
            sparse_service = chatbot.vector_search_service.sparse_service
        
        sparse_vectors = []
        for chunk in chunks:
            sparse_vec_dict = sparse_service.create_sparse_vector(chunk)
            # Convert from {token_id: weight} dict to Qdrant format {indices: [...], values: [...]}
            if sparse_vec_dict:
                indices = [int(k) for k in sparse_vec_dict.keys()]
                values = [float(v) for v in sparse_vec_dict.values()]
                sparse_vec = {"indices": indices, "values": values}
            else:
                sparse_vec = {"indices": [], "values": []}
            sparse_vectors.append(sparse_vec)
        logger.info(f"✅ Generated {len(sparse_vectors)} sparse vectors")
        
        # Step 4: Insert into Qdrant
        logger.info("💾 Inserting into Qdrant...")
        document_id = str(uuid.uuid4())
        points = []
        
        for i, (chunk, embedding, sparse_vec) in enumerate(zip(chunks, embeddings, sparse_vectors)):
            # Prepend title to chunk text if provided (so title is searchable in content)
            chunk_text = chunk
            if request.title:
                chunk_text = f"{request.title}\n\n{chunk}"
            
            point = {
                "id": str(uuid.uuid4()),
                "vector": {
                    "content": embedding  # Dense vector
                },
                "sparse_vector": {
                    "content_sparse": sparse_vec  # Sparse vector for hybrid search
                },
                "payload": {
                    "text": chunk_text,  # Include title in chunk text for searchability
                    "type": "custom",
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            # Add optional title if provided (also store in metadata)
            if request.title:
                point["payload"]["title"] = request.title
            
            points.append(point)
        
        # Use collection from config instead of hardcoded value
        from config import Config as ConfigClass
        collection_name = ConfigClass.QDRANT_COLLECTION
        qdrant_url = f"http://localhost:6333/collections/{collection_name}/points"
        qdrant_payload = {"points": points}
        
        qdrant_response = requests.put(qdrant_url, json=qdrant_payload, timeout=60)
        if qdrant_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Qdrant insertion failed: {qdrant_response.text}")
        
        logger.info(f"✅ Successfully inserted {len(points)} points into Qdrant")
        
        return CustomDocumentUploadResponse(
            success=True,
            message="Document uploaded and processed successfully",
            chunks_created=len(chunks),
            points_inserted=len(points),
            document_id=document_id,
            metadata={
                "content_type": request.content_type,
                "title": request.title,
                "total_characters": len(extracted_text),
                "chunking_algorithm": "word-count",
                "chunking_params": chunk_result.get('metadata', {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Custom document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Share conversation endpoints
@app.post("/share/conversation", response_model=ShareConversationResponse)
@app.post("/api/share/conversation", response_model=ShareConversationResponse)  # Also support /api/ prefix
async def share_conversation(request: ShareConversationRequest):
    """
    Share a conversation by storing it server-side and generating a shareable URL.
    
    Validates conversation data, generates a unique share ID, stores it as JSON,
    and returns a shareable URL.
    """
    try:
        # Validate conversation has messages
        if not request.history or len(request.history) == 0:
            raise HTTPException(status_code=400, detail="Cannot share empty conversation")
        
        # Limit conversation size (max 100 messages)
        if len(request.history) > 100:
            raise HTTPException(status_code=400, detail="Conversation too large (max 100 messages)")
        
        # Generate unique share ID
        share_id = str(uuid.uuid4())
        
        # Create share data
        share_data = {
            "id": share_id,
            "title": request.title,
            "timestamp": request.timestamp,
            "history": [msg.dict() for msg in request.history],
            "context": request.context,
            "created_at": int(datetime.now().timestamp() * 1000),
            "expires_at": int((datetime.now() + timedelta(days=30)).timestamp() * 1000)  # 30 days expiration
        }
        
        # Save to file
        shared_conv_dir = "shared_conversations"
        os.makedirs(shared_conv_dir, exist_ok=True)
        file_path = os.path.join(shared_conv_dir, f"{share_id}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(share_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Shared conversation created: {share_id} ({len(request.history)} messages)")
        
        # Generate share URL
        share_url = f"/test?share={share_id}"
        
        return ShareConversationResponse(
            share_id=share_id,
            share_url=share_url,
            expires_at=share_data["expires_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to share conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to share conversation: {str(e)}")

@app.get("/share/{share_id}", response_model=SharedConversationResponse)
@app.get("/api/share/{share_id}", response_model=SharedConversationResponse)  # Also support /api/ prefix
async def get_shared_conversation(share_id: str):
    """
    Retrieve a shared conversation by its share ID.
    
    Returns the conversation data or 404 if not found or expired.
    """
    try:
        # Validate share_id format (UUID)
        try:
            uuid.UUID(share_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid share ID format")
        
        # Load from file
        file_path = os.path.join("shared_conversations", f"{share_id}.json")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Shared conversation not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
        
        # Check expiration
        current_time = int(datetime.now().timestamp() * 1000)
        if share_data.get("expires_at") and current_time > share_data["expires_at"]:
            # Delete expired share
            os.remove(file_path)
            raise HTTPException(status_code=410, detail="Shared conversation has expired")
        
        # Convert history back to ChatMessage objects
        history = [ChatMessage(**msg) for msg in share_data["history"]]
        
        return SharedConversationResponse(
            id=share_data["id"],
            title=share_data["title"],
            timestamp=share_data["timestamp"],
            history=history,
            context=share_data.get("context"),
            created_at=share_data["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve shared conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve shared conversation: {str(e)}")

@app.get("/share/{share_id}/preview")
@app.get("/api/share/{share_id}/preview")  # Also support /api/ prefix
async def get_shared_conversation_preview(share_id: str):
    """
    Get a preview of a shared conversation (metadata only).
    
    Returns minimal information: title, message count, date.
    """
    try:
        # Validate share_id format
        try:
            uuid.UUID(share_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid share ID format")
        
        # Load from file
        file_path = os.path.join("shared_conversations", f"{share_id}.json")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Shared conversation not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
        
        # Check expiration
        current_time = int(datetime.now().timestamp() * 1000)
        if share_data.get("expires_at") and current_time > share_data["expires_at"]:
            os.remove(file_path)
            raise HTTPException(status_code=410, detail="Shared conversation has expired")
        
        return {
            "id": share_data["id"],
            "title": share_data["title"],
            "message_count": len(share_data["history"]),
            "created_at": share_data["created_at"],
            "timestamp": share_data["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve preview: {str(e)}")


# Sparse vector generation endpoint
class SparseVectorRequest(BaseModel):
    text: str = Field(..., description="Text to generate sparse vector for")

class SparseVectorResponse(BaseModel):
    sparse_vector: Dict[str, float] = Field(..., description="Sparse vector as {token_id: weight}")

@app.post("/generate_sparse_vector", response_model=SparseVectorResponse)
async def generate_sparse_vector(request: SparseVectorRequest):
    """Generate BM25 sparse vector for text using already-loaded model"""
    try:
        # Use the vector search service's sparse vector service
        sparse_vec = chatbot.vector_search_service.sparse_service.create_sparse_vector(request.text)
        return SparseVectorResponse(sparse_vector=sparse_vec)
    except Exception as e:
        logger.error(f"Error generating sparse vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure uvicorn to use httptools HTTP implementation
    # httptools has higher body size limits than h11 (default)
    # Note: For very large uploads (>50MB), consider using nginx as reverse proxy
    import uvicorn.config
    config = uvicorn.Config(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        limit_max_requests=10000,
        limit_concurrency=1000,
        http="httptools",  # Use httptools which has higher body size limits
    )
    server = uvicorn.Server(config)
    server.run()

