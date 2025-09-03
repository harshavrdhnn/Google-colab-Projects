# FastAPI Service

A simple FastAPI service with health check and sum calculation endpoints.

## Features

- **GET /health**: Returns service status
- **POST /sum**: Calculates the sum of a list of numbers
- Full type hints and Pydantic model validation
- OpenAPI documentation available at `/docs`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

```bash
python main.py
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "ok"}`

### Sum Calculation
- **URL**: `/sum`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "numbers": [1, 2, 3, 4, 5]
  }
  ```
- **Response**:
  ```json
  {
    "sum": 15.0
  }
  ```

## Examples

### Using curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Sum calculation
curl -X POST "http://localhost:8000/sum" \
  -H "Content-Type: application/json" \
  -d '{"numbers": [1, 2, 3, 4, 5]}'
```

## Documentation

Once the service is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`