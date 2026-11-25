from pydantic import BaseModel, Field
class AnalyzeRequest(BaseModel):
    resume_id: str = Field(min_length=1, max_length=100)
    job_description: str = Field(min_length=40, max_length=30000)
