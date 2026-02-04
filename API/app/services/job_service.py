import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

# imported models
from my_tests.utility.test_utils import run_refined_single # ReFinED EL model
from app.schemas.models import JobStatus, CellResult, Candidate, CandidateType # internal class/models


logger = logging.getLogger(__name__)

# internal "database" for jobs
JOBS: Dict[str, Dict[str, Any]] = {}


class JobService:
    @staticmethod
    def create_job(
            header: List[str],
            rows: List[dict],
            target_column: str,
            top_k: int
    ):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id": job_id,
            "status": JobStatus.created,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "header": header,
            "target_column": target_column,
            "top_k": top_k,
            "rows": rows,
            "result": None,
            "error": None,
            "current_row": 0,
            "total_rows": len(rows)
        }
        return job_id

    @staticmethod
    async def run_refined_task(job_id: str, model):
        job = JOBS.get(job_id)
        if not job: return

        # tries to update job status to running
        try:
            logger.info(f"Job {job_id}: Starting ReFinED processing")

            job["status"] = JobStatus.running
            job["updated_at"] = datetime.now(timezone.utc)

            # Extract data from the job object
            df = pd.DataFrame(job["rows"])
            target_column = job["target_column"]
            top_k = job["top_k"]

            # Progression tracking
            job["total_rows"] = len(df)
            job["current_row"] = 0

            final_results: List[CellResult] = []

            # Go through data rows and process, linking entities found
            for idx, row in df.iterrows():

                # check for cancellation
                if job["status"] == JobStatus.cancelled:
                    logger.info(f"Job {job_id}: Cancelled by user")
                    return

                # updates progress
                job["current_row"] = int(idx) + 1
                job["updated_at"] = datetime.now(timezone.utc)

                # extract mention
                mention = str(row[target_column])

                # runs the entity linking (using ReFinED)
                doc_spans_per_doc = run_refined_single([mention], model)[0]

                candidates_for_cell = []
                for span in doc_spans_per_doc:
                    predicted_entity = span.predicted_entity
                    predicted_qid = getattr(predicted_entity, "wikidata_entity_id", None) if predicted_entity else None

                    span_text = span.text
                    span_type = span.coarse_mention_type or span.coarse_type or "UNKNOWN"

                    # if candidates exists, go through and process
                    if span.candidate_entities:
                        for candidate, score in span.candidate_entities[:top_k]:
                            if isinstance(candidate, str):
                                c_qid = candidate
                                c_name = span_text
                            else:
                                c_qid = getattr(candidate, "wikidata_entity_id", None) or "null"
                                c_name = getattr(candidate, "wikipedia_entity_title", span_text)

                            # matching logic
                            is_match = (c_qid == predicted_qid) if (predicted_qid and c_qid != "null") else False

                            # map candidate to schema model
                            candidates_for_cell.append(
                                Candidate(
                                    id=str(c_qid),
                                    name=str(c_name),
                                    score=float(score),
                                    match=is_match,
                                    description=getattr(candidate, "description", ""),
                                    types=[CandidateType(id=span_type, name=span_type)] if is_match else []
                                )
                            )
                    else:
                        # if candidate does not exist, we add a "null candidate"
                        candidates_for_cell.append(
                            Candidate(
                                id="null",
                                name=span_text,
                                score=0.0,
                                match=False,
                                description="No candidates found by ReFinED",
                                types=[CandidateType(id="UNKNOWN", name="UNKNOWN")]
                            )
                        )
                # add CellResult to final list
                final_results.append(
                    CellResult(
                        row=int(idx),
                        col=target_column,
                        cell_id=f"{job_id}_{idx}",
                        mention=mention,
                        candidate_ranking=candidates_for_cell
                    )
                )
            job["result"] = final_results
            job["status"] = JobStatus.done
            job["updated_at"] = datetime.now(timezone.utc)
            logger.info(f"Job {job_id}: Completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job["status"] = JobStatus.failed
            job["error"] = str(e)
            job["updated_at"] = datetime.now(timezone.utc)