import os
import time
import requests
import yaml


from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
    Union,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

# from .utils import load_config

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ.get("OPENAI_API_BASE", "")
KEY = os.environ.get("OPENAI_API_KEY", "")
print('# api url: ', URL)
print('# api key: ', KEY)

# Qwen API Configuration (Idealab External - same as AgentNet)
AI_STUDIO_TOKEN = os.getenv("AI_STUDIO_TOKEN", "f0df527cb24da64c5e3d1618512bac86")
AI_STUDIO_API_URL = os.getenv("AI_STUDIO_API_URL", "https://idealab-external.alibaba-inc.com/api/openai/v1/chat/completions")

# Aliyun DashScope API (alternative)
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

MAX_RETRIES = 3


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens
        
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
                )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break 

        return "" 


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000


# ================================ Qwen LLM (Idealab API) ================================

class QwenChat(LLM):
    """
    Qwen LLM using Idealab External API (same as AgentNet).
    Supports models like qwen3-max, qwen3-flash, qwen-turbo, qwen-plus.
    """
    
    def __init__(
        self, 
        model_name: str = "qwen3-flash",
        use_idealab: bool = True,
    ):
        """
        Initialize Qwen Chat.
        
        Args:
            model_name: Model to use (qwen3-max, qwen3-flash, qwen-turbo, qwen-plus)
            use_idealab: Use Idealab API (True) or Aliyun DashScope (False)
        """
        super().__init__(model_name=model_name)
        self.use_idealab = use_idealab
        
        if use_idealab:
            self.api_url = AI_STUDIO_API_URL
            self.api_token = AI_STUDIO_TOKEN
        else:
            self.api_url = DASHSCOPE_API_URL
            self.api_token = ALIYUN_API_KEY
        
        print(f"QwenChat initialized with model={model_name}, use_idealab={use_idealab}")
    
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        """Call Qwen API."""
        # Add delay to avoid rate limiting (increased to 3s to reduce 429 errors)
        time.sleep(3)
        
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "model": self.model_name,
                    "messages": messages_dict,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                
                if stop_strs:
                    data["stop"] = stop_strs
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=120
                )
                response.raise_for_status()
                
                try:
                    result = response.json()
                except Exception as json_err:
                    print(f"JSON parse error. Response text: {response.text[:500]}")
                    raise ValueError(f"Invalid JSON response: {json_err}")
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
                else:
                    raise ValueError(f"Invalid response format: {result}")
                    
            except requests.exceptions.JSONDecodeError as e:
                print(f"Qwen API JSON decode error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                return ""
            except Exception as e:
                error_msg = str(e)
                # Try to get more error details
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_msg += f" - Response: {e.response.text[:500]}"
                    except:
                        pass
                print(f"Qwen API call attempt {attempt + 1} failed: {error_msg}")
                if attempt == MAX_RETRIES - 1:
                    print(f"All API attempts failed.")
                    return ""
                time.sleep(10 * (attempt + 1))  # Progressive backoff: 10s, 20s, 30s
        
        return ""


# ================================ Convenience Functions ================================

def get_qwen_response(
    system_prompt: str,
    query_prompt: str,
    model_name: str = "qwen3-flash",
    use_idealab: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Convenience function to call Qwen API directly.
    Similar to AgentNet's get_qwen_flash_response.
    
    Args:
        system_prompt: System message
        query_prompt: User query
        model_name: Model to use
        use_idealab: Use Idealab API
        temperature: Generation temperature
        max_tokens: Maximum tokens
        
    Returns:
        Model response text
    """
    llm = QwenChat(model_name=model_name, use_idealab=use_idealab)
    messages = [
        Message("system", system_prompt),
        Message("user", query_prompt)
    ]
    return llm(messages, temperature=temperature, max_tokens=max_tokens)


def main():
    llm = QwenChat(
        model_name="qwen-turbo",    # 也可以试 qwen3-max / qwen-plus
        use_idealab=True,           # 用 Idealab External
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Please explain what a transformer model is in one paragraph.")
    ]

    print("Calling Qwen API...")
    response = llm(
        messages,
        temperature=0.2,
        max_tokens=512,
    )

    print("\n===== Qwen Response =====")
    print(response)
    print("=========================")

if __name__ == "__main__":
    main()
