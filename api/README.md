# ReFinED API Service

This folder contains the **FastAPI** backend for the ReFinED entity linking model. 

It is designed to process tables and text strings, transforming them into **Koala-UI** compatible JSON.

---

### Model & Resource Requirements

To ensure high speed and accuracy for tables, it is configured with the following:

* **Model**: `wikipedia_model_with_numbers` — Optimized for tables containing dates and quantities.
* **Entity Set**: `wikidata` — Provides wide coverage (~33M entities) beyond standard Wikipedia.
* **Pre-computed Embeddings**: `use_precomputed=True`.
    * **Fast Inference**: Uses a local vector index for instant entity lookups.
    * **Hardware**: highly recommended having at least **32GB RAM** and an **SSD**.

---

### Troubleshooting & Tuning
* **Slow Startup**: The first load can take a short while, as the model maps the Wikidata index.
* **Memory Issues**: If you experience crashes (OOM), set `use_precomputed=False` or switch to `entity_set="wikipedia"` to reduce the memory usage.
* **Docker Users**: Ensure Docker is allowed sufficient memory to run with docker, should ideally be **28GB of RAM** in Settings > Resources.

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
