# ReFinED API Service

This folder contains the **FastAPI** backend for the ReFinED entity linking model. 

It is designed to process tables and text strings, transforming them into **Koala-UI** compatible JSON.

---

### Ecosystem Compatibility

This API is designed to be **interchangeable with [Crocodile](https://github.com/enRichMyData/crocodile)**. 

* **Interface Alignment**: The endpoints, request schemas, and response structures are modeled after Crocodile.
* **Seamless Exchange**: This service can replace Crocodile in existing pipelines, allowing users to upgrade to ReFinED's neural entity linking without changing integration logic.
* **Multipart Support**: Fully implements chunked upload logic for large-scale data ingestion.

#### API Endpoint Mapping
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/jobs` | `POST` | Initialize a job (Inline or Multipart) |
| `/jobs/{id}` | `GET` | Poll job status and metadata |
| `/jobs/{id}/parts` | `POST` | Upload data chunks (Multipart mode) |
| `/jobs/{id}/finalize`| `POST` | Signal end of upload and trigger AI processing |
| `/jobs/{id}/results` | `GET` | Retrieve Koala-formatted results |
| `/jobs/{id}/cancel`  | `POST` | Terminate a running or queued job |

---

### Job Lifecycle
To ensure compatibility with the Koala-UI state machine, jobs transition through the following states:



1.  **Ingesting**: (Multipart only) Accepting data chunks.
2.  **Queued**: Data received; waiting for the model worker.
3.  **Running**: Model is actively performing entity linking.
4.  **Done**: Results are ready for retrieval.
5.  **Error/Cancelled**: Terminal states for failed or aborted jobs.

---

### Key Features
* **Background Jobs**: Processes rows asynchronously to prevent HTTP timeouts on large tables.
* **Koala-UI Compatibility**: Output JSON matches the requirements of the [Koala-UI frontend](https://github.com/enRichMyData/koala_ui).
* **Device Adaptive**: Supports **GPU (CUDA)** and **CPU** (with optimized autocast).
* **CORS Enabled**: Configured to allow requests from frontend services (e.g., Koala-UI) out of the box.

---

### Folder Structure
* **`app/endpoints/`**: API route definitions for linking, jobs, and dataset listing.
* **`app/services/`**: The `JobService` logic for state management and background tasks.
* **`app/utility/`**: Model loading (`load_model`) and inference wrappers.
* **`app/schemas/`**: Pydantic models for Koala-format validation.
* **`app/main.py`**: API entry point and middleware configuration.
