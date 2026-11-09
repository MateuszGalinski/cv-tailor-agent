from pydantic import BaseModel, Field

class ScrapedOffer(BaseModel):
    """Use when you have the full job offer text."""
    title: str | None = None
    company: str | None = None
    text: str = Field(min_length=300, description="Full offer text as it appears on the page")

class ScrapeFailed(BaseModel):
    """Use when the page is not a job offer or the text cannot be retrieved."""
    reason: str = Field(description="e.g. login wall, offer expired, 404, captcha")