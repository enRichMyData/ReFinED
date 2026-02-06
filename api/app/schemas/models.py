from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field



class LinkRequest(BaseModel):
    """
    Used for "Link Single Text"
    """
    text: str = Field(..., example="James Cameron directed Titanic.")
    top_k: int = 5

# tracking progress
class JobStatus(str, Enum):
    created = "created"     # 0
    ingesting = "ingesting" # 1
    queued = "queued"       # 2
    running = "running"     # 3
    done = "done"           # 4
    failed = "failed"       # 5
    cancelled = "cancelled" # 6

class RowCells(BaseModel):
    row_id: Optional[Union[int, str]] = None
    cells: List[Any]

RowInput = Union[RowCells, Dict[str, Any]]

class JobCreateRequest(BaseModel):
    mode: Literal["inline", "multipart"] = "inline"
    header: List[str]
    rows: Optional[List[RowInput]] = None
    link_columns:  List[str]
    top_k: int = Field(default=5, ge=1, le=100)
    config: Dict[str, Any] = Field(default_factory=dict)
    total_parts: Optional[int] = Field(default=None, ge=1)
    total_rows: Optional[int] = Field(default=None, ge=0)
    table_name: Optional[str] = None

class JobIngestInfo(BaseModel):
    expected_parts: Optional[int] = None
    expected_rows: Optional[int] = None
    received_parts: int = 0
    received_rows: int = 0
    completed_at: Optional[datetime] = None

class JobProgressInfo(BaseModel):
    part_number: int = 0
    row_index: int = 0

class JobResultsInfo(BaseModel):
    segments: int = 0
    cells: int = 0

class JobErrorInfo(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None

# for job status API
class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    mode: Literal["inline", "multipart"] = "inline"
    created_at: datetime
    updated_at: datetime
    ingest: JobIngestInfo
    progress: JobProgressInfo
    results: JobResultsInfo
    error: Optional[JobErrorInfo] = None

class ResultsPage(BaseModel):
    ok: bool = True
    job_id: str
    cursor: Optional[str] = None
    next_cursor: Optional[str] = None
    results: List[Any]

class JobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str

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
