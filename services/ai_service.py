"""
AI Service Module
=================
Handles all interactions with the Ollama Cloud API for AI-powered chat responses.
Provides a modular interface for sending prompts and receiving AI completions,
with robust error handling and fallback mechanisms.

Uses Ollama's cloud-hosted models via the official Python SDK.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


class AIService:
    """
    Service class for interacting with the Ollama Cloud API.
    Uses the official ollama Python package with cloud hosting.
    """

    def __init__(self):
        """Initialize the Ollama client with API key from environment."""
        self.api_key = os.environ.get('OLLAMA_API_KEY', '')
        self.model_name = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b-cloud')
        self.fallback_models = self._parse_fallback_models()
        self.client = None

        if self.api_key:
            try:
                from ollama import Client
                self.client = Client(
                    host='https://api.ollama.com',
                    headers={'Authorization': f'Bearer {self.api_key}'}
                )
                logger.info(f"Ollama client initialized successfully with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
        else:
            logger.warning("OLLAMA_API_KEY not found in environment. AI service will be unavailable.")

    @staticmethod
    def _parse_fallback_models():
        """Parse the comma-separated fallback model list from the environment."""
        raw = os.environ.get('OLLAMA_FALLBACK_MODELS', 'qwen3:32b-cloud,llama3.1')
        return [m.strip() for m in raw.split(',') if m.strip()]

    def _model_chain(self):
        """Return the ordered list of models to try: primary first, then fallbacks."""
        chain = [self.model_name] if self.model_name else []
        for model in self.fallback_models:
            if model not in chain:
                chain.append(model)
        return chain

    def _ensure_client(self):
        """Ensure the Ollama client is available, re-initialize if needed."""
        if not self.client:
            self.api_key = os.environ.get('OLLAMA_API_KEY', '')
            if self.api_key:
                try:
                    from ollama import Client
                    self.client = Client(
                        host='https://api.ollama.com',
                        headers={'Authorization': f'Bearer {self.api_key}'}
                    )
                    self.model_name = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b-cloud')
                    self.fallback_models = self._parse_fallback_models()
                except Exception as e:
                    logger.error(f"Failed to reinitialize Ollama client: {e}")

    def get_chat_response(self, user_message, conversation_history=None, system_prompt=None):
        """
        Send a message to Ollama and get a response.

        Args:
            user_message: The user's input text.
            conversation_history: Optional list of prior messages for context.
            system_prompt: Optional system-level instructions for the AI.

        Returns:
            dict: Contains 'response' text and 'response_time' in seconds.
        """
        self._ensure_client()

        if not self.client:
            return {
                'response': "I'm sorry, the AI service is currently unavailable. "
                           "Please ensure the OLLAMA_API_KEY is configured in your .env file. "
                           "Get your API key at https://ollama.com/settings/keys",
                'response_time': 0
            }

        # Build the messages list for the Ollama API
        sys_instruction = system_prompt or (
            "You are an intelligent AI assistant built for an enterprise environment. "
            "You provide clear, professional, and helpful responses. "
            "You can help with analysis, writing, coding, research, and general questions. "
            "Format your responses with markdown when appropriate for better readability. "
            "Be concise but thorough."
        )

        # Build conversation messages
        messages = [
            {"role": "system", "content": sys_instruction}
        ]

        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                messages.append({
                    "role": "user",
                    "content": msg.get('prompt', '')
                })
                messages.append({
                    "role": "assistant",
                    "content": msg.get('response', '')
                })

        # Add the current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        start_time = time.time()
        last_error = None
        response_time = 0
        model_chain = self._model_chain()

        for idx, model in enumerate(model_chain):
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 4096,
                    }
                )

                response_text = response.message.content
                if not response_text or not response_text.strip():
                    raise ValueError("Empty response received from model")

                response_time = time.time() - start_time
                logger.info(f"AI response generated in {response_time:.2f}s using {model}")

                return {
                    'response': response_text,
                    'response_time': response_time,
                    'model_used': model
                }

            except Exception as e:
                last_error = e
                response_time = time.time() - start_time
                if idx + 1 < len(model_chain):
                    logger.warning(f"Model '{model}' failed ({e}). Falling back to: {model_chain[idx + 1]}")
                else:
                    logger.error(f"Model '{model}' failed ({e}). No fallback models remaining.")

        return {
            'response': f"I encountered an error processing your request. Please try again. Error: {last_error}",
            'response_time': response_time
        }

    def summarize_document(self, text, max_length=2000):
        """
        Generate an AI summary of a document's text content.

        Args:
            text: The extracted text content from the document.
            max_length: Maximum character length of text to send to the API.

        Returns:
            str: AI-generated summary of the document.
        """
        truncated_text = text[:max_length] if len(text) > max_length else text

        prompt = (
            f"Please provide a comprehensive summary of the following document. "
            f"Highlight the key points, main themes, and important details:\n\n"
            f"---\n{truncated_text}\n---"
        )

        result = self.get_chat_response(
            prompt,
            system_prompt=(
                "You are a document analysis expert. Provide clear, structured summaries "
                "with bullet points for key findings. Be thorough but concise."
            )
        )
        return result.get('response', 'Unable to generate summary.')

    def answer_document_question(self, text, question):
        """
        Answer a user's question about an uploaded document.

        Args:
            text: The extracted text content from the document.
            question: The user's question about the document.

        Returns:
            str: AI-generated answer based on the document content.
        """
        truncated_text = text[:3000] if len(text) > 3000 else text

        prompt = (
            f"Based on the following document content, please answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Document Content:\n---\n{truncated_text}\n---"
        )

        result = self.get_chat_response(
            prompt,
            system_prompt=(
                "You are a document analysis assistant. Answer questions accurately based "
                "on the provided document content. If the answer cannot be found in the document, "
                "say so clearly."
            )
        )
        return result.get('response', 'Unable to answer the question.')


# Singleton instance
ai_service = AIService()
