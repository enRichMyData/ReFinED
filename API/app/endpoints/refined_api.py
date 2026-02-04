from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# models and services
from app.schemas.models import JobStatus, JobStatusResponse, CellResult, Candidate
from app.services.job_service import JobService, JOBS
from app.config import settings
from my_tests.utility.test_utils import load_model


# router for the API, and ReFined Model
router = APIRouter()
model = load_model(device=settings.MODEL_DEVICE)

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


# Category: Instant Analysis (Synchronous)
# ============================================

@router.post("/link", tags=["Analysis"])
async def link_single_text(text: str, top_k: int = 5):
    """
    ### Instant Entity Linking
    Submit a single string and get results immediately.
    **Best for:** Real-time testing or single sentences.
    **Note:** For large tables, use the /jobs endpoint instead.
    """
    try:
        # Re-use the utility function used in the job service
        from my_tests.utility.test_utils import run_refined_single

        # runs the entity linker
        doc_spans_per_doc = run_refined_single([text], model)[0]

        results = []
        for span in doc_spans_per_doc:
            predicted = span.predicted_entity
            results.append({
                "mention": span.text,
                "predicted_qid": getattr(predicted, "wikidata_entity_id", None) if predicted else None,
                "confidence": float(span.candidate_entities[0][1]) if span.candidate_entities else 0.0,
                "type": span.coarse_mention_type or "UNKNOWN"
            })

        return {"text": text, "entities": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


# Category: Jobs: creating + status
# ======================================
@router.post("/jobs", status_code=202, tags=["Jobs"])
async def create_job(request: TableRequest, background_tasks: BackgroundTasks):
    """
    **Submit a table for Entity Linking**

    The endpoints starts a task done in the background. it will:
    1. Validate input table
    2. Queue job in memory
    3. Run ReFinED model in backgronud

    **Returns:** A 'job_id' used to track and fetch status/results
    """
    header = list(request.data[0].keys()) if request.data else []

    # create job for current request
    job_id = JobService.create_job(
        header=header,
        rows=request.data,
        target_column=request.target_column,
        top_k=request.top_k
    )

    # stores table name as metadata for identification
    JOBS[job_id]["table_name"] = request.table_name

    # offloads ReFinED entity linking execution to a background task
    background_tasks.add_task(JobService.run_refined_task, job_id, model)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", tags=["Jobs"], response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    ### Check Job Status
    Use this to see if your job is still 'running' or has 'completed'.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job.get("updated_at", job["created_at"]),
        progress={
            "row_index": job.get("current_row", 0),
            "total_rows": job.get("total_rows", 0)
        }
    )


@router.get("/jobs/{job_id}/results", tags=["Jobs"], response_model=List[CellResult])
async def get_job_results(job_id: str):
    """
    ### Get Final Results
    Once the status of a job is 'completed', use this endpoint to fetch the full linked data in the Koala JSON format.
    """
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # uses jobstatus
    if job["status"] != JobStatus.done:
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Current status {job['status']}")

    return job["result"]


# Category: Datasets (sort of from the "old" API)
# ============================================

@router.get("/datasets/default/tables", tags=["Datasets"])
def list_tables():
    """
    ### List All Tables
    Returns a summary of all EL jobs currently in memory
    """
    tables = []
    for j_id, j in JOBS.items():
        tables.append({
            "job_id": j_id,
            "table_name": j.get("table_name", "untitled"),
            "status": j["status"]  # This will now return "created", "running", etc.
        })

    return {"tables": tables}
