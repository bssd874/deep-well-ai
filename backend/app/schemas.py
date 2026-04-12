from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    total_records: int
    status: str
    input_type: Optional[str] = None
    detection_type: Optional[str] = None # To inform FE what was detected

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo