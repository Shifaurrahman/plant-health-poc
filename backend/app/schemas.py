from typing import Literal, Optional

from pydantic import BaseModel, Field


class CaseCreate(BaseModel):
    crop: Optional[str] = None
    region: Optional[str] = None
    symptom_duration: Optional[str] = None
    plant_part: Optional[str] = None
    recent_watering_or_treatment: Optional[str] = None
    consent_reuse_photos: bool = False


class CasePatch(BaseModel):
    crop: Optional[str] = None
    region: Optional[str] = None
    symptom_duration: Optional[str] = None
    plant_part: Optional[str] = None
    recent_watering_or_treatment: Optional[str] = None
    consent_reuse_photos: Optional[bool] = None


class FeedbackIn(BaseModel):
    turn_index: Optional[int] = None
    helpful: bool
    usefulness: Optional[int] = Field(default=None, ge=1, le=5)
    disputed: bool = False
    next_action_helped: Optional[bool] = None
    comment: Optional[str] = Field(default=None, max_length=1000)


class ReviewIn(BaseModel):
    turn_index: Optional[int] = None
    reviewer: str = Field(min_length=1, max_length=100)
    decision: Literal["confirmed", "corrected", "uncertain", "needs_lab_test"]
    confirmation_method: Literal[
        "expert_visual", "lab", "field_inspection", "none"
    ] = "none"
    confidence_basis: Optional[str] = Field(default=None, max_length=500)
    corrected_condition: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = Field(default=None, max_length=2000)
    publish_decision: Literal["exclude", "candidate_reference", "correction"] = "exclude"


class RecordPatch(BaseModel):
    review_status: Optional[Literal["pending", "approved", "retired"]] = None
    reviewed_by: Optional[str] = Field(default=None, max_length=100)
    symptoms: Optional[str] = None
    distinguishing_features: Optional[str] = None
    management_general: Optional[str] = None
    limitations: Optional[list[str]] = None
    next_step: Optional[str] = None
    alternatives: Optional[list[str]] = None
