"""
Base Agent class for all multi-agent orchestration patterns.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import openai
import asyncio
import time


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, role: str, api_key: str, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.role = role
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.execution_time = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process input data and return output."""
        pass
    
    async def _make_api_call(self, messages: list, temperature: float = 0.7) -> str:
        """Make an API call to OpenAI and track metrics."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            self.execution_time = time.time() - start_time
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.execution_time = time.time() - start_time
            raise Exception(f"API call failed for {self.name}: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get execution metrics for this agent."""
        return {
            "name": self.name,
            "role": self.role,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage
        }


class SimpleAgent(BaseAgent):
    """A simple agent implementation for basic tasks."""
    
    def __init__(self, name: str, role: str, api_key: str, system_prompt: str, model: str = "gpt-3.5-turbo"):
        super().__init__(name, role, api_key, model)
        self.system_prompt = system_prompt
    
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process input using the system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(input_data)}
        ]
        
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})
        
        result = await self._make_api_call(messages)
        return result
