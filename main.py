from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="FastAPI Service", description="A simple FastAPI service with health check and sum endpoints")


class NumberList(BaseModel):
    """Pydantic model for the request body containing a list of numbers."""
    numbers: List[float]


class HealthResponse(BaseModel):
    """Pydantic model for the health check response."""
    status: str


class SumResponse(BaseModel):
    """Pydantic model for the sum response."""
    sum: float


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint that returns the service status."""
    return HealthResponse(status="ok")


@app.post("/sum", response_model=SumResponse)
async def calculate_sum(data: NumberList) -> SumResponse:
    """Calculate the sum of a list of numbers."""
    total = sum(data.numbers)
    return SumResponse(sum=total)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)