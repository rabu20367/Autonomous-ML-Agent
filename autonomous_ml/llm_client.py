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
        """Parse and validate JSON response with robust error handling"""
        if not response or not response.strip():
            return {"error": "Empty response from LLM"}
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = json_match.group() if json_match else response.strip()
        
        # Clean up common JSON issues
        json_str = json_str.strip()
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        try:
            parsed = json.loads(json_str)
            # Validate that it's a dictionary
            if not isinstance(parsed, dict):
                return {"error": "Response is not a JSON object", "raw": response[:200]}
            return parsed
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Remove comments
                json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            return {
                "error": f"Invalid JSON response: {str(e)}", 
                "raw": response[:200],
                "attempted_parse": json_str[:200]
            }
