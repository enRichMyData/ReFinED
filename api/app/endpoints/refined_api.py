from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import Dict, Any, Optional

# models and services
from app.schemas.models import LinkRequest, JobStatus, JobStatusResponse, CellResult, Candidate, ResultsPage, JobCancelResponse, JobCreateRequest, RowCells
from app.utility.model_loader import load_model, run_refined_single
from app.services.job_service import JobService, JOBS
from app.config import settings


# router for the API, and ReFined Model
router = APIRouter()
model = load_model(
    device=settings.MODEL_DEVICE,
    model="wikipedia_model_with_numbers",   # <-- can be tweaked (e.g. fine-tuned model)
    entity_set="wikidata",                  # <-- can be tweaked
    use_precomputed=True
)


# Category: Instant Analysis (Synchronous)
# ============================================

@router.post("/link", tags=["Analysis"])
async def link_single_text(request: LinkRequest):
    """
    ### Instant Entity Linking
    Submit a single string and get results immediately.
    **Best for:** Real-time testing or single sentences.
    **Note:** For large tables, use the /jobs endpoint instead.
    """
    try:
        # runs the entity linker
        doc_spans_per_doc = run_refined_single([request.text], model)[0]

        results = []
        for span in doc_spans_per_doc:
            predicted = span.predicted_entity
            results.append({
                "mention": span.text,
                "predicted_qid": getattr(predicted, "wikidata_entity_id", None) if predicted else None,
                "confidence": float(span.candidate_entities[0][1]) if span.candidate_entities else 0.0,
                "type": span.coarse_mention_type or "UNKNOWN"
            })

        return {"text": request.text, "entities": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


# Category: Jobs: creating + status
# ======================================
@router.post("/jobs", status_code=202, tags=["Jobs"])
async def create_job(request: JobCreateRequest, background_tasks: BackgroundTasks):
    """
    **Submit a table for Entity Linking**

    The endpoints starts a task done in the background. it will:
    1. Validate input table
    2. Queue job in memory
    3. Run ReFinED model in backgronud

    **Returns:** A 'job_id' used to track and fetch status/results
    """
    header = request.header

    refined_rows = []
    if request.rows:
        for row in request.rows:
            if isinstance(row, RowCells):
                refined_rows.append(dict(zip(header, row.cells)))
            elif isinstance(row, dict):
                refined_rows.append(row)

    # link columns (like crocodile)
    target_col = request.link_columns[0] if request.link_columns else ""

    # create job for current request
    job_id = JobService.create_job(
        header=header,
        rows=refined_rows,
        target_column=target_col,
        top_k=request.top_k
    )

    # stores table name as metadata for identification
    if request.table_name:
        JOBS[job_id]["table_name"] = request.table_name

    # offloads ReFinED entity linking execution to a background task
    background_tasks.add_task(JobService.run_refined_task, job_id, model)

    return {
        "job_id": job_id,
        "status": JobStatus.queued,
        "mode": request.mode,
        "message": "Job accepted"
    }


@router.get("/jobs/{job_id}", tags=["Jobs"], response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    ### Check Job Status
    Returns the current state of the ReFinED task, including progress percentages.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        mode=job.get("mode", "inline"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        config_hash=job.get("config_hash", ""), # no config hash used in refined (yet?)
        ingest=job.get("ingest", {}),
        progress=job.get("progress", {}),
        results=job.get("results", {}),
        error=job.get("error")
    )

@router.post("/jobs/{job_id}:cancel", response_model=JobCancelResponse, tags=["Jobs"])
async def cancel_job(job_id: str):
    """
    ### Cancel a Job
    If a job is still in progress (not completed), you can cancel it to free up resources.
    """
    job = JobService.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobCancelResponse(
        job_id=job["job_id"],
        status=JobStatus.cancelled,
        message="Job cancelled successfully"
    )

@router.get("/jobs/{job_id}/results", tags=["Jobs"], response_model=ResultsPage)
async def get_job_results(
        job_id: str,
        cursor: Optional[str] = Query(None),
        limit: int = Query(100)
):
    """
    ### Get Final Results
    Once the status of a job is 'completed', use this endpoint to fetch the full linked data in the Koala JSON format.
    """
    job = JOBS.get(job_id)

    # 404 check
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # status check
    if job["status"] != JobStatus.done:
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Current status {job['status']}")

    # flattens koala format to a list of cellresults
    all_cell_results = []
    header = job["result"]["header"]

    for row_idx, row in enumerate(job["result"]["rows"]):
        for entity in row["linked_entities"]:
            col_idx = entity["idColumn"]

            # uses header for human-readable column name (e.g. country instead of 1)
            # column_name = header[col_idx] if col_idx < len(header) else str(col_idx)

            # create cellresult
            cell = CellResult(
                row=row_idx,
                col=col_idx,                    # crocodile uses col_idx, for readability, can use "column_name"
                cell_id=f"{row_idx}:{col_idx}",
                mention=row["data"][col_idx],
                candidate_ranking=entity["candidates"]
            )
            all_cell_results.append(cell)

    # pagination
    try:
        start = int(cursor) if (cursor and  cursor.isdigit()) else 0
    except (ValueError, TypeError):
        start = 0
    end = start + limit
    page_items = all_cell_results[start:end]
    next_cursor = str(end) if end < len(all_cell_results) else None

    return ResultsPage(
        ok=True,
        job_id=job_id,
        cursor=cursor,
        next_cursor=next_cursor,
        results=page_items
    )


# Category: Datasets (sort of from the "old" API)
# ============================================

@router.get("/datasets/default/tables", tags=["Datasets"])
def list_tables():
    """
    ### List All Tables
    Returns a summary of all EL jobs currently in memory
    """
    tables = []
    for job_id, job in JOBS.items():
        tables.append({
            "job_id": job_id,
            "table_name": job.get("table_name", "untitled"),
            "status": job["status"]
        })

    return {"tables": tables}


@router.get("/datasets", response_model=Dict[str, Any], tags=["Datasets"])
async def list_datasets(
        page: int = Query(1, ge=1, description="The page number (1-based)"),
        page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """
    ### List Datasets
    Uses a simple page numbering and sizes for pagination
    """
    # sort all jobs by newest
    all_jobs = sorted(JOBS.values(), key=lambda x: x["created_at"], reverse=True)
    total_count = len(all_jobs)
    total_pages = (total_count + page_size - 1) // page_size

    # calculation of start and end
    start = (page - 1) * page_size
    end = start + page_size

    # slice data
    page_items = all_jobs[start:end]

    # build pagination links
    next_page = f"/datasets?page={page + 1}&page_size={page_size}" if end < total_count else None
    prev_page = f"/datasets?page={page - 1}&page_size={page_size}" if page > 1 else None

    # transform jobs to koala
    datasets = []
    for job in page_items:
        datasets.append({
            "dataset_name": job.get("table_name", job.get("target_column", "Untitled Job")),
            "total_tables": 1,
            "total_rows": job.get("total_rows", 0),
            "created_at": job["created_at"].isoformat()
        })

    return {
        "data": datasets,
        "total": total_count,
        "page": page,
        "total_pages": total_pages,
        "pagination": {
            "next": next_page,
            "previous": prev_page
        }
    }




