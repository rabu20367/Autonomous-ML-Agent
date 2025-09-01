"""
Efficient LLM client with structured response generation
"""

import json
import re
from typing import Dict, Any, Optional
import openai
import anthropic
from pydantic import BaseModel, ValidationError

class StructuredLLMClient:
    """Unified LLM client with structured JSON response generation"""
    
    def __init__(self, provider: str, model: str, api_key: str, **kwargs):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == 'openai':
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_structured_response(self, prompt: str) -> Dict[str, Any]:
        """Generate structured JSON response with validation"""
        enhanced_prompt = self._enhance_for_json(prompt)
        raw_response = self._call_llm(enhanced_prompt)
        return self._parse_json_response(raw_response)
    
    def _enhance_for_json(self, prompt: str) -> str:
        """Enhance prompt to ensure JSON output"""
        return f"{prompt}\n\nIMPORTANT: Respond with valid JSON only. No additional text."
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM with error handling"""
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.kwargs.get('max_tokens', 4000),
                    temperature=self.kwargs.get('temperature', 0.1)
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.kwargs.get('max_tokens', 4000),
                    temperature=self.kwargs.get('temperature', 0.1),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            return f'{{"error": "LLM call failed: {str(e)}"}}'
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response"""
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = json_match.group() if json_match else response.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": response[:200]}
