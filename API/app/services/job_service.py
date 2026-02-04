import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

# imported models
from my_tests.utility.test_utils import run_refined_single # ReFinED EL model
from app.schemas.models import JobStatus, CellResult, Candidate, CandidateType # internal class/models

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

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

            # extracts data
            header = job["header"]
            target_column = job["target_column"]
            target_col_idx = header.index(target_column)
            top_k = job["top_k"]

            # Progression tracking
            raw_rows = job["rows"]
            job["total_rows"] = len(raw_rows)
            job["current_row"] = 0

            koala_rows = []

            coarse_counts = {}
            fine_counts = {}

            # Go through data rows and process, linking entities found
            for idx, row_dict in enumerate(raw_rows):

                # check for cancellation
                if job["status"] == JobStatus.cancelled:
                    logger.info(f"Job {job_id}: Cancelled by user")
                    return

                # updates progress
                job["current_row"] = int(idx) + 1
                job["updated_at"] = datetime.now(timezone.utc)

                # extract mention
                mention = str(row_dict.get(target_column, ""))

                # 1. runs the entity linking (using ReFinED)
                doc_spans_per_doc = run_refined_single([mention], model)[0]

                # 2. extract candidates
                candidates_for_cell = []
                for span in doc_spans_per_doc:
                    predicted_entity = span.predicted_entity
                    predicted_qid = getattr(predicted_entity, "wikidata_entity_id", None) if predicted_entity else None

                    span_text = span.text
                    span_type = span.coarse_mention_type or span.coarse_type or "OTHER"
                    coarse_counts[span_type] = coarse_counts.get(span_type, 0) + 1

                    # if candidates exists, go through and process
                    if span.candidate_entities:
                        has_match = False # fallback

                        for c_idx, (candidate, score) in enumerate(span.candidate_entities[:top_k]):
                            if isinstance(candidate, str):
                                c_qid = candidate
                                c_name = span_text
                            else:
                                c_qid = getattr(candidate, "wikidata_entity_id", None) or "null"
                                c_name = getattr(candidate, "wikipedia_entity_title", span_text)

                            # QID matching logic
                            is_match = (c_qid == predicted_qid) if (predicted_qid and c_qid != "null") else (c_idx == 0)
                            if is_match: has_match = True

                            # map candidate to schema model
                            candidate_obj = Candidate(
                                id=str(c_qid),
                                name=str(c_name),
                                score=float(score),
                                match=is_match,
                                description=getattr(candidate, "description", ""),
                                types=[CandidateType(id=span_type, name=span_type)] if is_match else []
                            )
                            candidates_for_cell.append(candidate_obj.model_dump())
                    else:
                        # if candidate does not exist, we add a "null candidate"
                        null_candidate = Candidate(
                            id="null",
                            name=span_text,
                            score=0.0,
                            match=False,
                            description="No candidates found by ReFinED",
                            types=[CandidateType(id="UNKNOWN", name="UNKNOWN")]
                        )
                        candidates_for_cell.append(null_candidate.model_dump())

                # 3. format koala row
                row_data_values = [str(row_dict.get(h, "")) for h in header]
                koala_rows.append({
                    "idRow": f"row_{idx}",
                    "data": row_data_values,
                    "linked_entities":[
                        {
                            "idColumn": target_col_idx,
                            "candidates": candidates_for_cell
                        }
                    ]
                })

            # Metadata processing
            # 1. gets most frequent coarse type
            most_frequent_coarse = max(coarse_counts, key=coarse_counts.get) if coarse_counts else "OTHER"

            # 2. basic LIT detection
            lit_map = {}
            for i, h in enumerate(header):
                if i == target_col_idx: continue
                if "year" in h.lower() or "date" in h.lower():
                    lit_map[str(i)] = "DATE"

            job["result"] = {
                "header": header,
                "rows": koala_rows,
                "status": "DONE",
                "classified_columns": {
                    "NE": {str(target_col_idx): most_frequent_coarse},
                    "LIT": lit_map
                },
                "column_types": {
                    str(target_col_idx): {
                        "types": [
                            {"id": most_frequent_coarse, "name": most_frequent_coarse, "count": len(koala_rows)}
                        ]
                    }
                }
            }

            job["status"] = JobStatus.done
            job["updated_at"] = datetime.now(timezone.utc)
            logger.info(f"Job {job_id}: Completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job["status"] = JobStatus.failed
            job["error"] = str(e)
            job["updated_at"] = datetime.now(timezone.utc)

