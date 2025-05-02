````markdown
# Project Setup Guide

## Prerequisites
- Python 3.8

## Installation Steps

1. **Create a virtual environment**:
   Run the following command to create a virtual environment named `.venv`:
   ```bash
   python3.8 -m venv .venv
````

2. **Install required dependencies**:
   Install the necessary dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to the backend directory**:
   Change to the `backend` directory where the main application resides:

   ```bash
   cd backend
   ```

4. **Start the application with Uvicorn**:
   Run the application using Uvicorn in development mode (`--reload` allows for auto-reloading):

   ```bash
   uvicorn main:app --reload
   ```

5. **Access API documentation**:
   Open your web browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to view the API documentation.

