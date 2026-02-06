# ReFinED API Service

This folder contains the **FastAPI** backend for the ReFinED entity linking model. 

It is designed to process tables and text strings, transforming them into **Koala-UI** compatible JSON.

---

### Folder Structure
* **`app/endpoints/`**: API route definitions for single linking, background jobs, and dataset listing.


* **`app/services/`**: The `JobService` logic for handling background tasks and in-memory job storage.


* **`app/utility/`**: Core model loading logic (`load_model`) and inference wrappers.
 

* **`app/schemas/`**: Pydantic models for data validation and Koala-format alignment.
 

* **`app/main.py`**: The API entry point, middleware configuration, and health checks.

---

### Key Features
* **Background Jobs**: Submit large tables via `/jobs` to avoid timeouts. The model processes rows in the background while you poll for progress.
  **(multi part upload WIP)**


* **Koala-UI Compatibility**: Output JSON is pre-formatted to match the requirements of the
  [Koala-UI frontend](https://github.com/enRichMyData/koala_ui).

* **Device Adaptive**: Supports both **GPU (CUDA)** for high-speed inference and **CPU** (with optimized autocast) for accessibility.


* **Interactive Docs**: Auto-generated Swagger UI available at the root URL (`/`) for real-time testing.

---

### Quick Start

**Run with Docker (recommended):**
From the project root:
```bash
docker compose up --build
