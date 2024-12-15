from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# Include the API routes
app.include_router(router)

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Gesture Validator API is running!"}