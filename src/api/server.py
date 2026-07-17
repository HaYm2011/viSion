from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.db.database import get_last_known_location
from src.llm.router import get_answer

app = FastAPI(title="Ambient Memory API")

class QueryRequest(BaseModel):
    text: str

@app.get("/status")
def status():
    return {"status": "running"}

@app.get("/last_seen/{object_name}")
def last_seen(object_name: str):
    location = get_last_known_location(object_name)
    if not location:
        raise HTTPException(status_code=404, detail="Object not found")
    return location

@app.post("/query")
def query(request: QueryRequest):
    answer = get_answer(request.text)
    return {"answer": answer}
