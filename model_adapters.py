"""
Model API Adapters - Unified interface for different LLM APIs
"""
import os
import json
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    """Base model adapter class"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize model adapter
        
        Args:
            model_name: Model name
            api_key: API key
        """
        self.model_name = model_name
        self.api_key = api_key
        self.total_tokens = 0
        self.request_count = 0
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """
        Generate model response
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            
        Returns:
            Model response text
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "model_name": self.model_name,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count
        }


class OpenAIAdapter(BaseModelAdapter):
    """OpenAI model adapter"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai library: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """Generate OpenAI response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.request_count += 1
            self.total_tokens += response.usage.total_tokens
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            # Check for quota errors
            if "insufficient_quota" in error_str or "429" in error_str:
                return f"ERROR_QUOTA: You exceeded your current quota. Please check your plan and billing details at https://platform.openai.com/account/billing"
            elif "rate_limit" in error_str.lower() or "429" in error_str:
                return f"ERROR_RATE_LIMIT: Rate limit exceeded. Please wait and try again later."
            else:
                return f"ERROR: {error_str}"


class ClaudeAdapter(BaseModelAdapter):
    """Claude model adapter"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Please install anthropic library: pip install anthropic")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """Generate Claude response"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            self.request_count += 1
            # Claude API token statistics
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text.strip()
        except Exception as e:
            error_str = str(e)
            # Check for quota/rate limit errors
            if "429" in error_str or "rate_limit" in error_str.lower():
                return f"ERROR_RATE_LIMIT: Rate limit exceeded. Please wait and try again later."
            elif "quota" in error_str.lower():
                return f"ERROR_QUOTA: Quota exceeded. Please check your billing at https://console.anthropic.com/"
            else:
                return f"ERROR: {error_str}"


class CustomAPIAdapter(BaseModelAdapter):
    """Custom API adapter (supports generic REST APIs)"""
    
    def __init__(self, model_name: str, api_url: str, api_key: Optional[str] = None, 
                 request_format: str = "openai"):
        """
        Initialize custom API adapter
        
        Args:
            model_name: Model name
            api_url: API endpoint URL
            api_key: API key
            request_format: Request format ("openai" or "custom")
        """
        super().__init__(model_name, api_key)
        self.api_url = api_url
        self.request_format = request_format
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests library: pip install requests")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """Generate custom API response"""
        try:
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            if self.request_format == "openai":
                # OpenAI-compatible format
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            else:
                # Custom format
                payload = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            
            response = self.requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            self.request_count += 1
            result = response.json()
            
            # Try to parse response
            if self.request_format == "openai":
                return result["choices"][0]["message"]["content"].strip()
            else:
                return result.get("text", result.get("response", str(result)))
                
        except Exception as e:
            return f"ERROR: {str(e)}"


class MockAdapter(BaseModelAdapter):
    """Mock adapter (for testing, does not call real APIs)"""
    
    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name, api_key=None)
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """Generate mock response"""
        self.request_count += 1
        
        # Simple mock logic
        if "sports" in prompt.lower() or "messi" in prompt.lower():
            return '{"label": "sports"}'
        elif "finance" in prompt.lower() or "stock" in prompt.lower():
            return '{"label": "finance"}'
        elif "positive" in prompt.lower() and "love" in prompt.lower():
            return '{"label": "positive"}'
        elif "2 + 2" in prompt:
            return '{"result": 4}'
        else:
            return '{"label": "unknown"}'


def create_adapter(adapter_type: str, model_name: str = None, 
                   api_key: Optional[str] = None, **kwargs) -> BaseModelAdapter:
    """
    Factory function: create model adapter
    
    Args:
        adapter_type: Adapter type ("openai", "claude", "custom", "mock")
        model_name: Model name
        api_key: API key
        **kwargs: Additional parameters
        
    Returns:
        Model adapter instance
    """
    if adapter_type == "openai":
        return OpenAIAdapter(model_name or "gpt-3.5-turbo", api_key)
    elif adapter_type == "claude":
        return ClaudeAdapter(model_name or "claude-3-sonnet-20240229", api_key)
    elif adapter_type == "custom":
        if "api_url" not in kwargs:
            raise ValueError("Custom API requires api_url parameter")
        return CustomAPIAdapter(model_name or "custom-model", kwargs["api_url"], api_key, 
                               kwargs.get("request_format", "openai"))
    elif adapter_type == "mock":
        return MockAdapter(model_name or "mock-model")
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
