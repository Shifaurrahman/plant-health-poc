from typing import Optional
from pydantic import BaseModel


class CaseCreate(BaseModel):
    crop: Optional[str] = None
    region: Optional[str] = None
    symptom_duration: Optional[str] = None
    plant_part: Optional[str] = None
