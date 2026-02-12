from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio

from Backend.llm_interpreter import SemanticInterpreter
from Backend.recommender import DJRecommendationEngine
from Backend.data.db_interface import DatabaseManager

app = FastAPI(title="AI DJ Curation API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced request models
class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None  # Current track, previous selections
    session_id: Optional[str] = None

class StructuredQuery(BaseModel):
    tags: List[str] = []
    vibe_descriptors: List[str] = []
    bpm_range: Optional[tuple[int, int]] = None
    energy_range: Optional[tuple[float, float]] = None
    key_compatibility: Optional[str] = None
    direction: Optional[str] = None  # "build", "maintain", "breakdown"
    exclude_tracks: List[str] = []

class CrateOperation(BaseModel):
    session_id: str
    tracks: List[str]
    sequence_order: List[int]
    metadata: Optional[Dict[str, Any]] = None

db_manager = DatabaseManager()

# Since we are using Supabase pgvector directly,
# we don't need a separate FAISS index.
embedding_index = None

semantic_interpreter = SemanticInterpreter()

# The recommendation engine now relies on db_manager for vector searches
recommendation_engine = DJRecommendationEngine(
    db_manager=db_manager,
    embedding_index=None
)
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI DJ Curation API",
        "version": "2.0.0"
    }

@app.post("/parse-intent")
async def parse_natural_language(query: NaturalLanguageQuery):
    """Convert natural language to structured query parameters"""
    try:
        # Get current context for better interpretation
        context = await db_manager.get_session_context(query.session_id) if query.session_id else None

        structured_query = await semantic_interpreter.interpret(
            query.query,
            context=context
        )

        return {
            "original_query": query.query,
            "structured_query": structured_query,
            "confidence": structured_query.get("confidence", 0.0),
            "interpretation_notes": structured_query.get("notes", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent parsing failed: {str(e)}")

@app.post("/intelligent-recommend")
async def intelligent_recommend(query: StructuredQuery, context_track_id: Optional[str] = None):
    """Get recommendations using structured parameters with DJ logic"""
    try:
        recommendations = await recommendation_engine.get_intelligent_recommendations(
            structured_query=query,
            context_track_id=context_track_id,
            apply_harmonic_weighting=True,
            apply_energy_flow=True,
            diversity_penalty=0.2
        )

        return {
            "recommendations": recommendations,
            "reasoning": recommendations.get("reasoning", {}),
            "compatibility_scores": recommendations.get("compatibility", {}),
            "pathway_visualization": recommendations.get("pathway_data", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/crate/operations")
async def manage_crate(operation: CrateOperation):
    """Handle crate operations with compatibility validation"""
    try:
        # Validate sequence compatibility
        compatibility_issues = await recommendation_engine.validate_sequence(
            operation.tracks,
            operation.sequence_order
        )

        # Persist to database
        await db_manager.update_crate(
            session_id=operation.session_id,
            tracks=operation.tracks,
            sequence=operation.sequence_order,
            validation_results=compatibility_issues
        )

        return {
            "success": True,
            "compatibility_issues": compatibility_issues,
            "sequence_score": compatibility_issues.get("overall_score", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crate operation failed: {str(e)}")

@app.get("/visualization/pathway")
async def get_pathway_visualization(from_track: str, to_tracks: List[str]):
    """Generate 3D pathway data for visualization"""
    try:
        pathway_data = await recommendation_engine.generate_pathway_visualization(
            source_track=from_track,
            target_tracks=to_tracks,
            max_intermediate_steps=3
        )

        return pathway_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pathway generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)