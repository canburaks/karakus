import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

from api.core.logger import log
from api.services.llama_index.config import get_llama_index_settings


class LlamaIndexService:
    """Service for interacting with LlamaIndex"""

    def __init__(self) -> None:
        self.settings = get_llama_index_settings()
        print(self.settings)
        print(f"{self.settings.SUPABASE_URL}/storage/v1/s3")
        log.info("Initialized LlamaIndex service")

    def read_files(
        self,
        input_files: List[str],
        recursive: bool = False,
        encoding: str = "utf-8",
        filename_as_id: bool = True,
        required_exts: Optional[List[str]] = None,
        file_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
        num_workers: int = 4,
    ) -> List[Document]:
        """Read specific files using SimpleDirectoryReader"""
        try:
            reader = SimpleDirectoryReader(
                input_files=input_files,
                recursive=recursive,
                encoding=encoding,
                filename_as_id=filename_as_id,
                required_exts=required_exts,
                file_metadata=file_metadata,
            )
            return reader.load_data(num_workers=num_workers)
        except Exception as e:
            log.error(f"Error reading files: {str(e)}")
            raise

    def read_directory(
        self,
        input_dir: str,
        recursive: bool = False,
        exclude: Optional[List[str]] = None,
        required_exts: Optional[List[str]] = None,
        num_files_limit: Optional[int] = None,
        encoding: str = "utf-8",
        filename_as_id: bool = True,
        file_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
        num_workers: int = 4,
    ) -> List[Document]:
        """Read files from a directory using SimpleDirectoryReader"""
        try:
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                recursive=recursive,
                exclude=exclude,
                required_exts=required_exts,
                num_files_limit=num_files_limit,
                encoding=encoding,
                filename_as_id=filename_as_id,
                file_metadata=file_metadata,
            )
            return reader.load_data(num_workers=num_workers)
        except Exception as e:
            log.error(f"Error reading directory: {str(e)}")
            raise

    def read_directory_iter(
        self,
        input_dir: str,
        recursive: bool = False,
        exclude: Optional[List[str]] = None,
        required_exts: Optional[List[str]] = None,
        num_files_limit: Optional[int] = None,
        encoding: str = "utf-8",
        filename_as_id: bool = True,
        file_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        """Iterate over files in a directory using SimpleDirectoryReader"""
        try:
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                recursive=recursive,
                exclude=exclude,
                required_exts=required_exts,
                num_files_limit=num_files_limit,
                encoding=encoding,
                filename_as_id=filename_as_id,
                file_metadata=file_metadata,
            )
            return reader.iter_data()
        except Exception as e:
            log.error(f"Error iterating directory: {str(e)}")
            raise

    def get_llm(self, **kwargs) -> LLM:
        """Get an LLM model"""
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        request_timeout = kwargs.get("request_timeout", 120000)

        if not isinstance(provider, str):
            raise ValueError("Provider is required and must be a string")
        if not isinstance(model, str):
            raise ValueError("Model is required and must be a string")

        if provider == "ollama":
            return Ollama(
                model=model,
                request_timeout=request_timeout,
                temperature=0.1,
                stop=["</s>", "Human:", "Assistant:"],
                format="json",
            )
        raise ValueError("Provider is not supported")

    def get_sllm(self, **kwargs):
        """Get a structured LLM model"""
        llm = self.get_llm(**kwargs)
        schema_cls = kwargs.get("schema")

        if not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
            raise ValueError("Schema must be a Pydantic BaseModel class")

        schema_json = schema_cls.model_json_schema()
        prompt = (
            "You are a precise JSON extractor. Extract information from the text below into a JSON object "
            "that follows this exact schema. Only output valid JSON, nothing else.\n\n"
            f"Schema: {json.dumps(schema_json, indent=2)}\n\n"
            "Text: {text}\n\n"
            "JSON output:"
        )

        def process_text(text: str) -> Any:
            response = llm.complete(prompt.format(text=text))
            try:
                # Try to parse the response as JSON
                json_str = response.text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("```", 1)[0]
                json_str = json_str.strip()

                # Parse and validate with Pydantic
                return schema_cls.model_validate_json(json_str)
            except Exception as e:
                log.error(f"Failed to parse JSON response: {e}")
                log.error(f"Raw response: {response.text}")
                raise ValueError(f"Failed to parse structured output: {e}")

        return process_text


# Global instance
_llama_index_instance: Optional[LlamaIndexService] = None


def get_llama_index_service() -> LlamaIndexService:
    """Get or create a LlamaIndex service instance"""
    global _llama_index_instance
    if _llama_index_instance is None:
        _llama_index_instance = LlamaIndexService()
    return _llama_index_instance
