from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field



class TableRequest(BaseModel):
    data: List[dict] = Field(
        ...,
        description="A list of row objects. Example: [{'name': 'Norway', 'pop': '5M'}]"
    )
    target_column: str = Field(
        ...,
        description="The name of the column containing the text you want to link."
    )
    top_k: int = Field(
        5,
        ge=1,
        description="Number of entity candidates to return per mention."
    )
    table_name: str = Field(
        "table1",
        description="A custom label for this table to help you find it later."
    )

class LinkRequest(BaseModel):
    """
    Used for "Link Single Text"
    """
    text: str = Field(..., example="James Cameron directed Titanic.")
    top_k: int = 5

# tracking progress
class JobStatus(str, Enum):
    created = "created"     # 0
    running = "running"     # 1
    done = "done"           # 2
    failed = "failed"       # 3
    cancelled = "cancelled" # 4

# Entity metadata
class CandidateType(BaseModel):
    id: str
    name: str

class Candidate(BaseModel):
    id: str
    name: str
    score: float
    match: bool = False
    types: List[CandidateType] = []
    description: Optional[str] = None

# Table result mapping
class CellResult(BaseModel):
    row: Union[int, str]
    col: Union[int, str]
    cell_id: str
    mention: str
    candidate_ranking: List[Candidate]

# for job status API
class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    progress: Dict[str, int] = Field(default_factory=lambda: {"row_index": 0})