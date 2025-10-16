# src/api/main.py

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

# Set the TFHUB_CACHE_DIR environment variable BEFORE importing tensorflow or tensorflow_hub
# This ensures BERT models are downloaded to and loaded from a persistent local directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
tfhub_cache_dir = os.path.join(current_dir, 'tfhub_cache')
os.environ['TFHUB_CACHE_DIR'] = tfhub_cache_dir

# Create the cache directory if it doesn't exist
if not os.path.exists(tfhub_cache_dir):
    os.makedirs(tfhub_cache_dir)
    logging.info(f"Created TFHUB_CACHE_DIR at: {tfhub_cache_dir}")
else:
    logging.info(f"Using existing TFHUB_CACHE_DIR: {tfhub_cache_dir}")

# Import the SkillAnalyzer (after setting the cache directory)
from src.analytics.skill_analyzer import SkillAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Skill Intelligence API",
    description="API for personalized skill recommendations and skill insights.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly include OPTIONS
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)
# Initialize SkillAnalyzer globally
skill_analyzer = SkillAnalyzer()

@app.on_event("startup")
async def startup_event():
    """Event handler that runs when the FastAPI application starts."""
    logging.info("API Startup: Loading necessary ML resources...")
    try:
        skill_analyzer.load_resources()
        logging.info("API Startup: All resources loaded successfully.")
    except RuntimeError as e:
        logging.error(f"API Startup Error: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load API resources: {e}")

# --- Pydantic Models for Request and Response ---

class SkillInput(BaseModel):
    skill_name: str = Field(..., description="The name of the skill to query.")

class SkillDetails(BaseModel):
    skill_name: str
    category: str
    skill_type: str
    description: Optional[str] = Field(None, description="A brief description of the skill.")
    difficulty_level_scaled: float
    learning_time_days: int
    popularity_score: float
    job_demand_score: float
    salary_impact_percent: int
    prerequisites: List[str]
    complementary_skills: List[str]
    market_trend: str
    industry_usage_text: str
    certification_available: bool
    future_relevance_score: float
    learning_resources_quality: float
    predicted_future_relevance_score: Optional[float] = None
    predicted_salary_impact_percent: Optional[float] = None
    predicted_job_demand_score: Optional[float] = None
    predicted_learning_time_days: Optional[float] = None
    predicted_certification_available: Optional[bool] = None

class SkillRecommendation(BaseModel):
    skill_name: str
    score: float = Field(..., description="Relevance score for the recommendation.")
    reason: str = Field(..., description="Explanation for why the skill is recommended.")

class UserProfile(BaseModel):
    current_skills: List[str] = Field(..., description="List of skills the user currently possesses.")
    goals: List[str] = Field(..., description="List of career or learning goals of the user.")

class RecommendationResponse(BaseModel):
    recommendations: List[SkillRecommendation]

class LearningRoadmapRequest(BaseModel):
    current_skills: List[str] = Field(..., description="Skills the user currently possesses.")
    learning_goals: List[str] = Field(..., description="Specific skills or roles the user wants to achieve.")
    desired_pace: Optional[str] = Field("moderate", description="Desired learning pace (e.g., 'fast', 'moderate', 'slow').")

class LearningRoadmapStep(BaseModel):
    skill_name: str
    estimated_days: int = Field(..., description="Estimated days to learn this skill.")
    reason: str = Field(..., description="Why this skill is included in the roadmap.")
    prerequisites: Optional[List[str]] = Field(None, description="Skills recommended to learn before this one.")
    complementary_skills: Optional[List[str]] = Field(None, description="Skills that complement this one.")

class LearningRoadmapResponse(BaseModel):
    roadmap: List[LearningRoadmapStep]
    overall_estimated_time_days: int
    message: str = "Personalized learning roadmap generated."

# --- New Pydantic Models for Peer/Mentor Matching ---
class MatchPeerRequest(BaseModel):
    user_skills_to_learn: List[str] = Field(..., description="Skills the user wants to learn.")
    user_skills_to_teach: List[str] = Field(..., description="Skills the user can teach.")
    preferred_learning_style: Optional[str] = Field(None, description="User's preferred learning style (e.g., 'visual', 'hands-on').")
    availability_preference: Optional[str] = Field(None, description="User's availability preference (e.g., 'weekends', 'evenings').")

class MatchedPeer(BaseModel):
    user_id: str = Field(..., description="Placeholder for actual user ID from your user management system.")
    name: str = Field(..., description="Name of the matched peer/mentor.")
    skills_can_teach: List[str] = Field(..., description="Skills this peer can teach relevant to user's needs.")
    skills_want_to_learn: List[str] = Field(..., description="Skills this peer wants to learn relevant to user's teaching.")
    compatibility_score: float = Field(..., description="Score indicating the match quality (0.0 to 1.0).")
    reason: str = Field(..., description="Explanation for why this peer is a good match.")

class MatchPeerResponse(BaseModel):
    matches: List[MatchedPeer]
    message: str = "Peer/mentor matching complete."

# --- New Pydantic Models for Skill Gap Identification ---
class GoalSkillGapsRequest(BaseModel):
    current_skills: List[str] = Field(..., description="Skills the user currently possesses.")
    target_goal: str = Field(..., description="Specific career or learning goal (e.g., 'become a Machine Learning Engineer').")

class SkillGap(BaseModel):
    skill_name: str
    importance_score: float = Field(..., description="Importance score for learning this skill to achieve the goal (0.0 to 1.0).")
    current_proficiency: Optional[float] = Field(None, description="User's current estimated proficiency (0.0 to 1.0).")
    recommended_learning_priority: str = Field(..., description="Priority for learning this skill (e.g., 'High', 'Medium', 'Low').")

class GoalSkillGapsResponse(BaseModel):
    skill_gaps: List[SkillGap]
    message: str = "Skill gaps identified for the target goal."

# --- New Pydantic Models for Learning Progress Analysis ---
class LearningProgressRequest(BaseModel):
    user_id: str = Field(..., description="ID of the user for whom to track progress.") # Assumes you have a user ID
    completed_activities: Optional[List[str]] = Field(None, description="List of completed courses/projects/certifications (e.g., 'Python Basics Course', 'E-commerce Project').")
    self_assessed_skills: Optional[Dict[str, float]] = Field(None, description="Self-assessed skill levels (e.g., {'Python': 0.7, 'SQL': 0.5}).")

class SkillGrowthMetric(BaseModel):
    skill_name: str
    initial_proficiency: Optional[float] = Field(None, description="Proficiency at start of tracking period.")
    current_proficiency: float = Field(..., description="Current estimated proficiency.")
    growth_percentage: float = Field(..., description="Percentage growth in proficiency.")
    trend: str = Field(..., description="e.g., 'increasing', 'steady', 'decreasing'")

class LearningProgressResponse(BaseModel):
    overall_progress_summary: str = Field(..., description="A high-level summary of the user's learning progress.")
    skill_growth_metrics: List[SkillGrowthMetric]
    next_steps_suggestions: List[str] = Field(..., description="Suggested next steps based on progress and remaining goals.")

# --- New Pydantic Models for Portfolio Optimization ---
class PortfolioOptimizeRequest(BaseModel):
    current_skills: List[str] = Field(..., description="Skills the user currently possesses.")
    completed_projects: Optional[List[str]] = Field(None, description="Names or descriptions of completed projects relevant to portfolio.")
    target_roles: Optional[List[str]] = Field(None, description="Target career roles for optimizing the portfolio (e.g., 'Data Scientist', 'Full-stack Developer').")

class OptimizedPortfolioElement(BaseModel):
    type: str = Field(..., description="e.g., 'skill', 'project'")
    name: str
    emphasis_reason: str = Field(..., description="Why this element should be emphasized in a portfolio.")
    relevance_score: float = Field(..., description="Relevance to target roles/brand (0.0 to 1.0).")

class PortfolioOptimizeResponse(BaseModel):
    optimized_portfolio_elements: List[OptimizedPortfolioElement]
    personal_branding_guidance: List[str] = Field(..., description="Tips for enhancing personal brand based on skills and goals.")
    message: str = "Portfolio optimization complete."

# --- New Pydantic Models for Market Skill Demand & Competitive Positioning ---
class MarketInsightsRequest(BaseModel):
    user_skills: List[str] = Field(..., description="Skills the user currently possesses.")
    target_industries: Optional[List[str]] = Field(None, description="Industries the user is interested in (e.g., 'Fintech', 'Healthcare').")
    target_roles: Optional[List[str]] = Field(None, description="Roles the user is interested in (e.g., 'Software Engineer', 'Product Manager').")

class SkillMarketData(BaseModel):
    skill_name: str
    demand_level: str = Field(..., description="e.g., 'High', 'Medium', 'Low'")
    average_salary_impact: Optional[float] = Field(None, description="Estimated average salary impact (e.g., 0.1 for 10% increase).")
    growth_trend: str = Field(..., description="e.g., 'increasing', 'stable', 'decreasing'")
    competitive_advantage_score: Optional[float] = Field(None, description="Score indicating how unique/valuable this skill combo is (0.0 to 1.0).")

class MarketInsightsResponse(BaseModel):
    market_overview_summary: str = Field(..., description="A summary of the market trends relevant to the user's skills and goals.")
    skill_market_data: List[SkillMarketData]
    positioning_suggestions: List[str] = Field(..., description="Suggestions to improve market positioning or find niche opportunities.")
# --- API Endpoints ---

@app.get("/api/v1/skills/details/{skill_name_query}", response_model=SkillDetails, summary="Get details and predictions for a specific skill")
async def get_skill_details(skill_name_query: str):
    """
    Retrieves detailed information and model predictions for a given skill using a GET request.
    Performs fuzzy and semantic matching if an exact skill name is not found.
    """
    try:
        skill_data = skill_analyzer.perform_skill_analysis(skill_name_query)
        return SkillDetails(**skill_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/skill/analyze", response_model=SkillDetails, summary="Analyze a single skill and get its predicted attributes")
async def analyze_skill(skill_input: SkillInput):
    """
    Provides a comprehensive analysis of an individual skill, including its predicted attributes and derived insights.
    Accepts the skill name in the request body.
    """
    try:
        skill_data = skill_analyzer.perform_skill_analysis(skill_input.skill_name)
        return SkillDetails(**skill_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/recommendations/skills", response_model=RecommendationResponse, summary="Get personalized skill recommendations with reasoning")
async def recommend_skills(user_profile: UserProfile):
    """
    Provides personalized skill recommendations based on the user's current skills and goals.
    Each recommendation includes a score and a reasoning explanation.
    """
    try:
        recommendations_list = skill_analyzer.recommend_skills(user_profile.dict())
        if not recommendations_list:
            return RecommendationResponse(recommendations=[], detail="No relevant recommendations found based on your profile.")
        return RecommendationResponse(recommendations=recommendations_list)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

# src/api/main.py

# ... (existing API Endpoints like get_skill_details, analyze_skill, recommend_skills)

@app.post("/api/v1/learning-roadmap")
async def generate_learning_roadmap(request: dict):
    """
    Generate an optimized learning roadmap with complementary skills integration.
    """
    try:
        # Extract parameters from request
        user_profile = request.get('user_profile', {})
        target_skills = request.get('target_skills', None)
        roadmap_length_weeks = request.get('roadmap_length_weeks', 12)
        skills_per_phase = request.get('skills_per_phase', 3)
        
        # Validate user_profile structure
        if not user_profile.get('current_skills'):
            user_profile['current_skills'] = []
        if not user_profile.get('goals'):
            user_profile['goals'] = []
            
        # Generate roadmap using your SkillAnalyzer
        roadmap = skill_analyzer.generate_learning_roadmap(
            user_profile=user_profile,
            target_skills=target_skills,
            roadmap_length_weeks=roadmap_length_weeks,
            skills_per_phase=skills_per_phase
        )
        
        return {
            "success": True, 
            "roadmap": roadmap,
            "message": "Learning roadmap generated successfully"
        }
        
    except ValueError as ve:
        logger.error(f"Validation error in learning roadmap: {ve}")
        return {
            "success": False, 
            "error": f"Validation error: {str(ve)}"
        }
    except Exception as e:
        logger.error(f"Error generating learning roadmap: {e}")
        return {
            "success": False, 
            "error": f"Failed to generate roadmap: {str(e)}"
        }

@app.post("/api/v1/learning-roadmap/preview")
async def preview_learning_roadmap(request: dict):
    """
    Preview what skills would be included in a roadmap without full generation.
    """
    try:
        user_profile = request.get('user_profile', {})
        target_skills = request.get('target_skills', None)
        
        # Get base recommendations
        recommendations = skill_analyzer.recommend_skills(user_profile)
        
        # Add target skills if provided
        preview_skills = []
        if target_skills:
            preview_skills.extend(target_skills)
            
        # Add top recommendations
        preview_skills.extend([rec['skill_name'] for rec in recommendations[:10]])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_preview = []
        for skill in preview_skills:
            if skill.lower() not in seen:
                unique_preview.append(skill)
                seen.add(skill.lower())
        
        return {
            "success": True,
            "preview_skills": unique_preview[:15],
            "total_recommendations": len(recommendations),
            "message": "Roadmap preview generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating roadmap preview: {e}")
        return {
            "success": False,
            "error": f"Failed to generate preview: {str(e)}"
        }

@app.post("/api/v1/user/skill-market-insights", response_model=MarketInsightsResponse, summary="Provide market insights on skill demand and competitive positioning")
async def post_market_skill_demand(request: MarketInsightsRequest):
    """
    Provides insights into the market demand for a user's skills and suggests strategies for competitive positioning.
    """
    try:
        # Conceptual Logic:
        # 1. For each user_skill, fetch market data (job_demand_score, future_relevance_score, salary_impact_percent)
        #    from skill_analyzer's dataset.
        # 2. Analyze against target_industries/target_roles to filter/prioritize.
        # 3. Calculate 'competitive_advantage_score' (e.g., unique combinations of skills, high demand + low supply).
        # 4. Generate overall market summary and positioning suggestions.

        # Placeholder/Mockup response for now
        market_data = []
        positioning_suggestions = []
        market_summary = "Based on your skills and interests: "

        if not request.user_skills:
            raise HTTPException(status_code=400, detail="User skills are required to get market insights.")

        for skill in request.user_skills:
            skill_details = skill_analyzer.perform_skill_analysis(skill) # Use existing analysis to get data
            if skill_details:
                demand = "High" if skill_details.get("job_demand_score", 0) > 0.7 else "Medium"
                salary_impact = skill_details.get("salary_impact_percent", 0) / 100.0 # Convert from percentage
                growth = skill_details.get("market_trend", "stable")
                
                # Simple competitive advantage: higher future relevance and job demand
                competitive_score = round(skill_details.get("future_relevance_score", 0.5) * 0.8 + skill_details.get("job_demand_score", 0.5) * 0.2, 2)

                market_data.append(SkillMarketData(
                    skill_name=skill,
                    demand_level=demand,
                    average_salary_impact=salary_impact,
                    growth_trend=growth,
                    competitive_advantage_score=competitive_score
                ))
                if demand == "High":
                    market_summary += f"Demand for {skill} is {demand}. "

        if not market_data:
            market_summary = "Could not find specific market data for your skills. "
            positioning_suggestions.append("Consider exploring foundational skills like Python or Data Analysis.")
        else:
            positioning_suggestions.append("Focus on developing a niche by combining highly demanded skills with complementary ones.")
            positioning_suggestions.append("Showcase your interdisciplinary skills in your portfolio.")
            market_summary += "Analyze the skill market data below for detailed insights."

        return MarketInsightsResponse(
            market_overview_summary=market_summary,
            skill_market_data=market_data,
            positioning_suggestions=positioning_suggestions
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Service error during market insights generation: {e}")

@app.post("/api/v1/peer-match")
async def peer_match(request: dict):
    """
    Accepts JSON payload:
    {
        "user_profile": { "want_to_learn": str, "can_teach": str, "goals": [...], "current_skills": [...] },
        "peer_profiles": [
            { "user_id": ..., "name": ..., "want_to_learn": str, "can_teach": str, "goals": [...], "current_skills": [...] },
             ...
        ]
    }
    Returns list of matched peers with detailed scores.
    """
    try:
        user_profile = request.get("user_profile")
        peer_profiles = request.get("peer_profiles", [])
        if not user_profile or not peer_profiles:
            return {"success": False, "error": "Missing user_profile or peer_profiles."}

        matches = skill_analyzer.match_peer(user_profile, peer_profiles)
        return {"success": True, "matches": matches}
    except Exception as e:
        logger.error(f"Peer match error: {e}")
        return {"success": False, "error": str(e)}
    

@app.post("/api/v1/user/market-position")
async def calculate_user_market_position(request: dict):
    """
    Calculate user's comprehensive market position score and provide actionable insights.
    
    Expected payload:
    {
        "current_skills": ["Python", "React", "SQL"],
        "goals": ["become a full-stack developer", "work in fintech"],
        "interests": ["web development", "financial technology"]
    }
    """
    try:
        user_profile = request.get("user_profile", {})
        
        # Validate input
        if not user_profile.get('current_skills'):
            return {
                "success": False, 
                "error": "current_skills are required to calculate market position"
            }
        
        # Calculate market position
        position_data = skill_analyzer.calculate_user_market_position(user_profile)
        
        return {
            "success": True,
            "market_position": position_data,
            "message": "Market position calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error calculating market position: {e}")
        return {
            "success": False,
            "error": f"Failed to calculate market position: {str(e)}"
        }

if __name__ == "__main__":
    # Get the port from environment variable (Render sets $PORT)
    port = int(os.environ.get("PORT", 8000))  # fallback for local testing
    uvicorn.run(
        "src.api.main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )