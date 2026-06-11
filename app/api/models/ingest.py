from pydantic import BaseModel


class IngestRequest(BaseModel):
    text: str
    source: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    inserted: int
    duration_ms: int


class LCIngestRequest(BaseModel):
    text: str
    source: str


class DeleteResponse(BaseModel):
    source: str
    deleted: int


class StatsResponse(BaseModel):
    collection: str
    row_count: int
    fields: list[str]
