"""
Visual Encoder Module for G-Memory++
Uses Qwen-VL API (qwen3-vl-plus) for GUI grounding and visual understanding.
Based on AgentNet's Idealab API pattern.
"""

import os
import base64
import time
import json
import re
import requests
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


# ================================ API Configuration ================================

# Qwen VL API Configuration (Idealab External) - same as AgentNet
AI_STUDIO_TOKEN = os.getenv("AI_STUDIO_TOKEN", "f0df527cb24da64c5e3d1618512bac86")
AI_STUDIO_API_URL = os.getenv("AI_STUDIO_API_URL", "https://idealab-external.alibaba-inc.com/api/openai/v1/chat/completions")

# Alternative: Aliyun DashScope API
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

MAX_RETRIES = 3


# ================================ Visual Prompts ================================

UI_DESCRIBE_PROMPT = """Analyze this screenshot and describe the UI elements that are visible.
Focus on:
1. Main interactive elements (buttons, links, input fields)
2. Their approximate locations (top-left, center, bottom-right, etc.)
3. Any text labels visible on these elements
4. The overall layout and structure of the interface

Be concise but thorough.
"""

UI_LOCATE_PROMPT = """Look at this screenshot and find the UI element matching this description:
"{element_description}"

Provide the approximate bounding box coordinates as [x1, y1, x2, y2] where:
- (x1, y1) is the top-left corner
- (x2, y2) is the bottom-right corner
- Coordinates are normalized to 0-1 range (percentage of image size)

If the element is not found, respond with "NOT_FOUND".
"""

UI_ACTION_PROMPT = """Based on the instruction: "{instruction}"

And the current screenshot, determine:
1. What UI element should be interacted with
2. What action should be performed (click, type, scroll, etc.)
3. The approximate location of that element

Format your response as:
ELEMENT: [description of the element]
ACTION: [action type]
LOCATION: [x, y] in normalized coordinates (0-1)
"""


# ================================ Data Classes ================================

@dataclass
class VisualContext:
    """Represents visual context from a screenshot or UI state."""
    
    # Image data
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    
    # Extracted information
    description: str = ""
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Visual embeddings (for similarity search)
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    source: str = ""  # "screenshot", "ui_dump", etc.
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "description": self.description,
            "ui_elements": self.ui_elements,
            "source": self.source,
            "timestamp": self.timestamp,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VisualContext":
        return VisualContext(
            image_path=data.get("image_path"),
            description=data.get("description", ""),
            ui_elements=data.get("ui_elements", []),
            source=data.get("source", ""),
            timestamp=data.get("timestamp"),
        )


@dataclass
class UIElement:
    """Represents a UI element detected in a screenshot."""
    
    element_type: str  # "button", "text_field", "icon", "link", etc.
    label: Optional[str] = None
    description: str = ""
    
    # Bounding box (normalized 0-1)
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    
    # Center point (normalized 0-1)
    center: Optional[Tuple[float, float]] = None
    
    # Confidence score
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "label": self.label,
            "description": self.description,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": self.confidence,
        }


# ================================ Qwen VL API Client ================================

class QwenVLAPIClient:
    """
    API client for Qwen-VL models using Idealab External API (same as AgentNet).
    Supports qwen3-vl-plus and other vision-language models.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-vl-plus",
        use_idealab: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize Qwen-VL API client.
        
        Args:
            model_name: Model to use ("qwen3-vl-plus", "qwen-vl-max", etc.)
            use_idealab: Use Idealab API (True) or Aliyun DashScope API (False)
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.use_idealab = use_idealab
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if use_idealab:
            self.api_url = AI_STUDIO_API_URL
            self.api_token = AI_STUDIO_TOKEN
        else:
            self.api_url = DASHSCOPE_API_URL
            self.api_token = ALIYUN_API_KEY
        
        print(f"QwenVLAPIClient initialized with model={model_name}, use_idealab={use_idealab}")
    
    def _image_to_base64(self, image: Union[str, "Image.Image"]) -> str:
        """Convert image to base64 string."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image processing")
        
        if isinstance(image, str):
            # Load from path
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def call_vl_api(
        self,
        prompt: str,
        image: Optional[Union[str, "Image.Image"]] = None,
        image_base64: Optional[str] = None,
        system_prompt: str = "You are a helpful visual assistant that analyzes UI screenshots."
    ) -> str:
        """
        Call Qwen-VL API with image and text prompt.
        
        Args:
            prompt: Text prompt/instruction
            image: PIL Image or path to image file
            image_base64: Base64 encoded image (alternative to image)
            system_prompt: System message
            
        Returns:
            Model response text
        """
        # Prepare image data
        if image is not None:
            img_b64 = self._image_to_base64(image)
        elif image_base64 is not None:
            img_b64 = image_base64
        else:
            img_b64 = None
        
        # Build messages in OpenAI vision format
        if img_b64:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            user_content = prompt
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Add delay to avoid rate limiting (like AgentNet)
        time.sleep(0.5)
        
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                
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
                print(f"Qwen VL API JSON decode error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                return ""
            except Exception as e:
                print(f"Qwen VL API call attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"All API attempts failed.")
                    return ""
                time.sleep(3)
        
        return ""


# ================================ Visual Encoder Base ================================

class VisualEncoderBase:
    """Base class for visual encoders."""
    
    def encode_image(self, image: Union[str, "Image.Image"]) -> np.ndarray:
        """Encode an image to a feature vector."""
        raise NotImplementedError
    
    def describe_ui(self, image: Union[str, "Image.Image"]) -> str:
        """Generate a description of the UI in an image."""
        raise NotImplementedError
    
    def locate_element(
        self, 
        image: Union[str, "Image.Image"], 
        element_description: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Locate a UI element by description. Returns bbox or None."""
        raise NotImplementedError
    
    def get_action_target(
        self, 
        image: Union[str, "Image.Image"], 
        instruction: str
    ) -> Dict[str, Any]:
        """Determine action target from instruction and image."""
        raise NotImplementedError


# ================================ Qwen-VL Encoder ================================

class QwenVLEncoder(VisualEncoderBase):
    """
    Visual encoder using Qwen-VL API (qwen3-vl-plus).
    Uses Idealab External API like AgentNet.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-vl-plus",
        use_idealab: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize Qwen-VL encoder.
        
        Args:
            model_name: Model to use (default: qwen3-vl-plus)
            use_idealab: Use Idealab API (True) or Aliyun DashScope (False)
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.api_client = QwenVLAPIClient(
            model_name=model_name,
            use_idealab=use_idealab,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f"QwenVLEncoder initialized with {model_name}")
    
    def _load_image(self, image: Union[str, "Image.Image"]) -> "Image.Image":
        """Load image from path or return PIL Image."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image processing")
        
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")
    
    def encode_image(self, image: Union[str, "Image.Image"]) -> np.ndarray:
        """
        Encode image to feature vector.
        Uses a simple hash-based approach since API doesn't provide embeddings.
        For more sophisticated embeddings, use a local embedding model.
        """
        pil_image = self._load_image(image)
        # Resize to small fixed size and flatten as pseudo-embedding
        small = pil_image.resize((32, 32))
        arr = np.array(small).flatten().astype(np.float32)
        # Normalize
        return arr / (np.linalg.norm(arr) + 1e-8)
    
    def describe_ui(self, image: Union[str, "Image.Image"]) -> str:
        """Generate description of UI elements in the image."""
        return self.api_client.call_vl_api(
            prompt=UI_DESCRIBE_PROMPT,
            image=image,
            system_prompt="You are an expert UI analyst. Describe the UI elements visible in the screenshot concisely."
        )
    
    def locate_element(
        self,
        image: Union[str, "Image.Image"],
        element_description: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Locate a UI element by description."""
        prompt = UI_LOCATE_PROMPT.format(element_description=element_description)
        
        response = self.api_client.call_vl_api(
            prompt=prompt,
            image=image,
            system_prompt="You are a UI element locator. Find elements and return their bounding boxes."
        )
        
        return self._parse_bbox(response)
    
    def _parse_bbox(self, response: str) -> Optional[Tuple[float, float, float, float]]:
        """Parse bounding box from model response."""
        if "NOT_FOUND" in response.upper():
            return None
        
        # Try to extract coordinates
        patterns = [
            r'\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]',
            r'(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    coords = tuple(float(x) for x in match.groups())
                    return coords
                except ValueError:
                    continue
        
        return None
    
    def get_action_target(
        self,
        image: Union[str, "Image.Image"],
        instruction: str
    ) -> Dict[str, Any]:
        """Determine action target from instruction and image."""
        prompt = UI_ACTION_PROMPT.format(instruction=instruction)
        
        response = self.api_client.call_vl_api(
            prompt=prompt,
            image=image,
            system_prompt="You are a GUI action planner. Determine what action to take based on the instruction."
        )
        
        result = {
            "element": None,
            "action": None,
            "location": None,
            "raw_response": response
        }
        result.update(self._parse_action_response(response))
        
        return result
    
    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """Parse action response from model."""
        result = {}
        
        # Parse ELEMENT
        element_match = re.search(r'ELEMENT:\s*(.+?)(?:\n|$)', response)
        if element_match:
            result["element"] = element_match.group(1).strip()
        
        # Parse ACTION
        action_match = re.search(r'ACTION:\s*(\w+)', response)
        if action_match:
            result["action"] = action_match.group(1).strip().lower()
        
        # Parse LOCATION
        location_match = re.search(r'LOCATION:\s*\[?\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\]?', response)
        if location_match:
            try:
                x, y = float(location_match.group(1)), float(location_match.group(2))
                result["location"] = (x, y)
            except ValueError:
                pass
        
        return result
    
    def chat_with_image(
        self,
        image: Union[str, "Image.Image"],
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        General chat with image capability.
        
        Args:
            image: Screenshot or UI image
            question: Question about the image
            system_prompt: Optional custom system prompt
        
        Returns:
            Model response
        """
        return self.api_client.call_vl_api(
            prompt=question,
            image=image,
            system_prompt=system_prompt or "You are a helpful visual assistant."
        )


# ================================ Fallback Encoder ================================

class FallbackVisualEncoder(VisualEncoderBase):
    """
    Fallback encoder when no vision model is available.
    Uses text descriptions or placeholder values.
    """
    
    def __init__(self):
        print("FallbackVisualEncoder initialized (no vision capabilities)")
    
    def encode_image(self, image: Union[str, Any]) -> np.ndarray:
        """Return placeholder embedding."""
        return np.zeros(3072, dtype=np.float32)  # 32x32x3 flattened
    
    def describe_ui(self, image: Union[str, Any]) -> str:
        """Return placeholder description."""
        return "Visual description not available (no vision model configured)"
    
    def locate_element(
        self,
        image: Union[str, Any],
        element_description: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Cannot locate without vision model."""
        return None
    
    def get_action_target(
        self,
        image: Union[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """Return empty result."""
        return {
            "element": None,
            "action": None,
            "location": None,
            "raw_response": "Vision model not available"
        }


# ================================ Visual Encoder Factory ================================

def create_visual_encoder(
    encoder_type: str = "qwen-vl",
    model_name: str = "qwen3-vl-plus",
    use_idealab: bool = True,
    **kwargs
) -> VisualEncoderBase:
    """
    Factory function to create appropriate visual encoder.
    
    Args:
        encoder_type: "qwen-vl", "fallback", or "auto"
        model_name: Model to use (default: qwen3-vl-plus)
        use_idealab: Use Idealab API (True) or Aliyun DashScope (False)
        **kwargs: Additional arguments for the encoder
    
    Returns:
        VisualEncoderBase instance
    """
    if encoder_type == "fallback":
        return FallbackVisualEncoder()
    
    if encoder_type == "auto":
        # Check if we have API token configured
        if AI_STUDIO_TOKEN or ALIYUN_API_KEY:
            encoder_type = "qwen-vl"
        else:
            print("Warning: No API token configured, falling back to FallbackVisualEncoder")
            return FallbackVisualEncoder()
    
    if encoder_type == "qwen-vl":
        return QwenVLEncoder(
            model_name=model_name,
            use_idealab=use_idealab,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
    
    return FallbackVisualEncoder()


# ================================ Convenience Functions ================================

def get_qwen_vl_response(
    prompt: str,
    image: Optional[Union[str, "Image.Image"]] = None,
    image_base64: Optional[str] = None,
    model_name: str = "qwen3-vl-plus",
    system_prompt: str = "You are a helpful visual assistant.",
    use_idealab: bool = True,
) -> str:
    """
    Convenience function to call Qwen-VL API directly.
    Similar to AgentNet's get_qwen_flash_response but with vision support.
    
    Args:
        prompt: Text prompt
        image: PIL Image or path (optional)
        image_base64: Base64 encoded image (optional)
        model_name: Model to use
        system_prompt: System message
        use_idealab: Use Idealab API
        
    Returns:
        Model response text
    """
    client = QwenVLAPIClient(model_name=model_name, use_idealab=use_idealab)
    return client.call_vl_api(
        prompt=prompt,
        image=image,
        image_base64=image_base64,
        system_prompt=system_prompt
    )
