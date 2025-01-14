# Personal RAG Application (In-development)

A FastAPI-based Retrieval-Augmented Generation (RAG) application that integrates multiple AI services and vector embeddings for document processing and AI interactions.

## Technology Stack

### Core Frameworks & Libraries
- **FastAPI**: Main web framework for building the API
- **Pydantic**: Data validation using Python type annotations
- **Uvicorn**: ASGI server implementation
- **httpx**: Async HTTP client for making API calls
- **Rich**: Advanced terminal output formatting and logging

### AI & ML Integration
- **HuggingFace**: Integration for transformer models and embeddings
  - Support for both API and local model implementations
  - Sentence transformers for embeddings
- **OpenRouter**: Multi-model AI platform integration
  - Access to various LLMs (Claude-3, GPT-4, Gemini Pro, Mistral, Qwen)
  - Embeddings generation capabilities

### Database & Authentication
- **Supabase**: 
  - Authentication and user management
  - Vector storage and similarity search
  - Token management

### Document Processing
- **MarkItDown**: PDF parsing and text extraction
- Support for both PDF and text file processing

### Payment Integration
- **LemonSqueezy**: Payment processing and subscription management
  - Token-based usage tracking
  - Webhook handling for subscription events

## Key Features
- User authentication and authorization
- File upload and processing (PDF and text)
- Vector embeddings generation and storage
- Token-based access control
- AI model interaction with multiple providers
- Subscription management
- Rich logging and error handling

## Authentication & Security
- JWT-based authentication
- Token balance verification
- Webhook signature verification
- Rate limiting and usage tracking

## Project Structure
```text
.
├── api/
│   ├── core/
│   │   ├── logger.py
│   │   └── parser.py
│   ├── services/
│   │   ├── huggingface/
│   │   ├── openrouter/
│   │   ├── supabase/
│   │   └── lemonsqueezy/
│   ├── middlewares/
│   └── routers/
```


