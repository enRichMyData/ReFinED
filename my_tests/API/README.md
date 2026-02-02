# ReFinED FastAPI Backend (Koala-UI Compatible)

This folder provides a **FastAPI-based backend service** for the entity linking tool **ReFinED**.
It exposes Koala-UI–compatible endpoints for uploading tables, running entity linking, and retrieving results.

The service is designed to support experimentation, evaluation, and visualization of entity linking results, and serves as the backend integration layer for **Koala-UI**.

---

## Features

- Entity linking using **ReFinED**
- Koala-UI compatible JSON responses
- In-memory dataset and table storage
- REST API for:
  - Uploading tables
  - Listing datasets
  - Listing tables
  - Fetching processed tables
  - Checking table status

---

## Requirements

- Python 3.10+
- ReFinED installed and working
- GPU support recommended (CPU also works)

### Python dependencies

```bash
pip install fastapi uvicorn pandas requests pydantic
