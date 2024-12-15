from typing import List, Dict, Optional, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class NewsSource(BaseModel):
    """Represents a news source with reliability metrics"""
    name: str = Field(..., description="Name of the news outlet")
    url: HttpUrl = Field(..., description="Base URL of the news source")
    reliability_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Source reliability score from 0 to 1"
    )
    bias_rating: Optional[str] = Field(
        None,
        description="Political/editorial bias rating"
    )
    domain_authority: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Domain authority score"
    )

class NewsQuote(BaseModel):
    """Represents a quote extracted from a news article"""
    text: str = Field(..., description="The quoted text")
    speaker: Optional[str] = Field(
        None,
        description="Person or entity being quoted"
    )
    context: Optional[str] = Field(
        None,
        description="Context surrounding the quote"
    )
    timestamp: Optional[str] = Field(
        None,
        description="When the quote was made, if available"
    )
    sentiment_score: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Sentiment score of the quote from -1 to 1"
    )

class NewsEntity(BaseModel):
    """Represents a named entity mentioned in the article"""
    name: str = Field(..., description="Name of the entity")
    type: Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "OTHER"] = Field(
        ...,
        description="Type of the named entity"
    )
    sentiment: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Sentiment score for mentions of this entity"
    )
    mentions: List[str] = Field(
        default_factory=list,
        description="List of contextual mentions of this entity"
    )
    importance_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Relative importance of this entity in the article"
    )

class NewsMetadata(BaseModel):
    """Metadata about the news article"""
    publication_date: Optional[datetime] = Field(
        None,
        description="When the article was published"
    )
    last_updated: Optional[datetime] = Field(
        None,
        description="When the article was last updated"
    )
    author: Optional[str] = Field(
        None,
        description="Article author(s)"
    )
    section: Optional[str] = Field(
        None,
        description="News section or category"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Article tags or keywords"
    )

class NewsAnalysis(BaseModel):
    """Analysis results for the news article"""
    overall_sentiment: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Overall sentiment score"
    )
    objectivity_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Objectivity score"
    )
    topics: List[str] = Field(
        ...,
        description="Main topics discussed"
    )
    key_points: List[str] = Field(
        ...,
        description="Key points from the article"
    )
    bias_indicators: List[str] = Field(
        default_factory=list,
        description="Potential bias indicators found"
    )

class NewsArticle(BaseModel):
    """Complete representation of a news article with analysis"""
    title: str = Field(..., description="Article headline")
    url: HttpUrl = Field(..., description="Article URL")
    content: str = Field(..., description="Full article text")
    summary: str = Field(..., description="Article summary")
    
    # Metadata and source information
    metadata: NewsMetadata = Field(..., description="Article metadata")
    source: NewsSource = Field(..., description="Source information")
    
    # Content analysis
    quotes: List[NewsQuote] = Field(
        default_factory=list,
        description="Quotes extracted from the article"
    )
    entities: List[NewsEntity] = Field(
        default_factory=list,
        description="Named entities found in the article"
    )
    analysis: NewsAnalysis = Field(..., description="Article analysis results")
    
    # Additional fields for processing
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this article was processed"
    )
    version: str = Field(
        default="1.0",
        description="Schema version"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Major Tech Company Announces New AI Initiative",
                "url": "https://example.com/news/tech-ai-initiative",
                "content": "Article content...",
                "summary": "Brief summary of the article...",
                "metadata": {
                    "publication_date": "2024-03-20T15:30:00Z",
                    "author": "Jane Doe",
                    "section": "Technology"
                },
                "source": {
                    "name": "Tech News Daily",
                    "url": "https://example.com",
                    "reliability_score": 0.85
                }
            }
        } 