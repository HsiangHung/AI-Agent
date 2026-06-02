from pydantic import BaseModel, Field
from typing import Literal

class TelemetryLog(BaseModel):
    event_type: Literal["account_creation", "login", "checkout"]
    is_fraud: int = Field(description="1 for fraud, 0 for legitimate")
    fraud_type: Literal["synthetic_identity", "bot_card_testing", "none"]
    
    # Device Signals
    device_os: str
    screen_resolution: str
    battery_level: float = Field(description="Battery level between 0.0 and 1.0")
    
    # Behavioral Signals
    typing_speed_wpm: int = Field(description="Words per minute")
    mouse_trajectory_entropy: float = Field(description="0.0 (perfectly straight) to 1.0 (highly erratic human movement)")
    time_to_complete_form_sec: float
    
    # Network
    ip_isp: str = Field(description="e.g., Comcast, AWS, DigitalOcean")
