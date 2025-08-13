import uuid
import asyncio
import json
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Body, Query, Request, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import desc, func, and_
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base, get_db
from config import settings
from saml_auth import (
    init_saml_auth, prepare_fastapi_request, create_access_token, verify_token,
    extract_user_info_from_saml, is_valid_university_email, determine_user_role
)
from models import (
    Task, GeneratedPrompt, PromptEvaluation, PromptSuggestion, PromptAnalysis, 
    LibraryPrompt, PromptStar, User, UserPrompt, LibrarySubmission
)
from celery_worker import (
    create_initial_prompt_task,
    evaluate_prompt_task,
    generate_suggestions_task,
    analyze_and_tag_task,
    analyze_submission_task
)


# --- Logging Middleware ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            logger.info(f"Request Body for {request.url.path}: {body.decode()}")
            
            async def receive():
                return {"type": "http.request", "body": body}
            
            new_request = Request(request.scope, receive)
            response = await call_next(new_request)
        else:
            response = await call_next(request)
        
        return response

# --- Security ---
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    
    # Verify JWT token
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        return None
    
    return user

async def require_admin(
    admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Require admin privileges"""
    # Allow admin key access (for external admin tools)
    if admin_key == settings.ADMIN_KEY:
        return True
    
    # Allow logged-in admin users
    if current_user and current_user.is_admin:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required"
    )

# --- Pydantic Models ---

class UserCreate(BaseModel):
    email: str
    username: str
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    student_id: Optional[str] = None
    affiliation: Optional[str] = None
    is_admin: bool = False
    is_student: bool = False
    is_faculty: bool = False
    is_staff: bool = False

class UserUpdate(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    student_id: Optional[str] = None
    affiliation: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    is_student: Optional[bool] = None
    is_faculty: Optional[bool] = None
    is_staff: Optional[bool] = None

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime
    prompt_count: int = 0

class LibrarySubmissionRequest(BaseModel):
    prompt_text: str
    submission_notes: Optional[str] = None

class LibrarySubmissionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    prompt_text: str
    submission_notes: Optional[str] = None
    status: str
    admin_notes: Optional[str] = None
    reviewed_by: Optional[uuid.UUID] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    user_email: Optional[str] = None

class AdminReviewRequest(BaseModel):
    action: str  # "approve" or "reject"
    admin_notes: Optional[str] = None

class InitialGenerationRequest(BaseModel):
    user_needs: str
    deepeval_context: Optional[str] = None

class EvaluationRequest(BaseModel):
    prompt: str
    user_needs: str
    retrieved_content: Optional[str] = None
    ground_truths: Optional[str] = None

class SuggestionRequest(BaseModel):
    current_prompt: str
    user_comments: Optional[str] = None
    retrieved_content: Optional[str] = None

class AnalysisRequest(BaseModel):
    prompt_text: str
    targeted_context: Optional[str] = None

class ToggleStarRequest(BaseModel):
    user_id: str

class TaskCreationResponse(BaseModel):
    task_id: uuid.UUID
    status_url: str

class BaseTaskResponse(BaseModel):
    task_id: uuid.UUID
    task_type: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class GeneratedPromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_needs: str
    initial_prompt: str

class EvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    original_prompt: str
    improved_prompt: str
    improvement_summary: str
    bias_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    alignment_score: Optional[float] = None

class SuggestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    category: str
    title: str
    description: str
    priority_score: int

class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    summary: str
    tags: List[str]
    quality_indicators: Dict[str, Any]
    category_analysis: Dict[str, str]

class TaskStatusResponse(BaseTaskResponse):
    result: Optional[Union[GeneratedPromptResponse, EvaluationResponse, List[SuggestionResponse], AnalysisResponse, "LibraryPromptResponse"]] = None

class LibraryPromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    text: str
    created_at: datetime
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    stars: int = 0
    is_starred_by_user: bool = False

class UserPromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    prompt_text: str
    task_type: str
    created_at: datetime
    summary: Optional[str] = None
    tags: Optional[List[str]] = None

class AuthSessionResponse(BaseModel):
    isAuthenticated: bool
    user: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: str
    username: str
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    student_id: Optional[str] = None
    affiliation: Optional[str] = None
    is_active: bool
    is_admin: bool
    is_student: bool = False
    is_faculty: bool = False
    is_staff: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime
    prompt_count: int = 0


# --- FastAPI App ---
app = FastAPI(
    title="Advanced Prompt Engineering Service",
    description="A comprehensive API for generating, evaluating, and optimizing prompts with admin management.",
    version="3.0.0"
)

app.add_middleware(RequestLoggingMiddleware)
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    # Don't auto-create tables since we're using migrations
    pass

# --- Authentication Endpoints ---

@app.get("/auth/session", response_model=AuthSessionResponse, tags=["Authentication"])
async def get_session(current_user: Optional[User] = Depends(get_current_user)):
    """Get current user session"""
    if current_user:
        return AuthSessionResponse(
            isAuthenticated=True,
            user={
                "id": str(current_user.id),
                "email": current_user.email,
                "username": current_user.username,
                "name": current_user.full_name or f"{current_user.first_name} {current_user.last_name}".strip(),
                "is_admin": current_user.is_admin,
                "is_student": current_user.is_student,
                "is_faculty": current_user.is_faculty,
                "is_staff": current_user.is_staff
            }
        )
    else:
        return AuthSessionResponse(isAuthenticated=False, user=None)

# Mock Authentication Endpoint (for development/testing)
@app.post("/auth/mock/login", tags=["Authentication"])
async def mock_login(
    email: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db)
):
    """Mock authentication endpoint for testing (simulates SAML)"""
    
    # Mock user data based on email patterns
    mock_users = {
        "student@ufl.edu": {
            "email": "student@ufl.edu",
            "username": "student1",
            "first_name": "John",
            "last_name": "Student",
            "student_id": "12345678",
            "affiliation": "student",
            "is_student": True,
            "is_faculty": False,
            "is_staff": False,
            "is_admin": False
        },
        "faculty@ufl.edu": {
            "email": "faculty@ufl.edu",
            "username": "faculty1",
            "first_name": "Jane",
            "last_name": "Professor",
            "student_id": None,
            "affiliation": "faculty",
            "is_student": False,
            "is_faculty": True,
            "is_staff": False,
            "is_admin": False
        },
        "admin@ufl.edu": {
            "email": "admin@ufl.edu",
            "username": "admin1",
            "first_name": "Admin",
            "last_name": "User",
            "student_id": None,
            "affiliation": "staff",
            "is_student": False,
            "is_faculty": False,
            "is_staff": True,
            "is_admin": True
        },
        "demo@example.com": {
            "email": "demo@example.com",
            "username": "demo_user",
            "first_name": "Demo",
            "last_name": "User",
            "student_id": None,
            "affiliation": "student",
            "is_student": True,
            "is_faculty": False,
            "is_staff": False,
            "is_admin": False
        }
    }
    
    # Get mock user data or create default student
    if email in mock_users:
        user_data = mock_users[email]
    else:
        # Default student data for any UFL email
        if email.endswith('@ufl.edu'):
            username = email.split('@')[0]
            user_data = {
                "email": email,
                "username": username,
                "first_name": username.capitalize(),
                "last_name": "User",
                "student_id": f"mock{hash(email) % 100000000:08d}",
                "affiliation": "student",
                "is_student": True,
                "is_faculty": False,
                "is_staff": False,
                "is_admin": False
            }
        else:
            raise HTTPException(
                status_code=403,
                detail="Access restricted to university accounts (@ufl.edu)"
            )
    
    # Create or update user in database
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if user:
        # Update existing user
        user.last_login = datetime.utcnow()
        for field, value in user_data.items():
            if field != "email":  # Don't update email
                setattr(user, field, value)
        user.full_name = f"{user.first_name} {user.last_name}".strip()
    else:
        # Create new user
        user = User(
            **user_data,
            full_name=f"{user_data['first_name']} {user_data['last_name']}".strip(),
            last_login=datetime.utcnow(),
            is_active=True
        )
        db.add(user)
    
    await db.commit()
    await db.refresh(user)
    
    # Create JWT token
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "is_admin": user.is_admin
    }
    access_token = create_access_token(data=token_data)
    
    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "name": user.full_name,
            "is_admin": user.is_admin,
            "is_student": user.is_student,
            "is_faculty": user.is_faculty,
            "is_staff": user.is_staff
        }
    }

@app.get("/auth/saml/login", tags=["Authentication"])
async def saml_login(request: Request):
    """Initiate SAML login"""
    req = prepare_fastapi_request(request)
    auth = init_saml_auth(req)
    
    # Redirect to Identity Provider
    sso_url = auth.login()
    return RedirectResponse(url=sso_url, status_code=302)

@app.post("/auth/saml/callback", tags=["Authentication"])
async def saml_callback(request: Request, SAMLResponse: str = Form(...), db: AsyncSession = Depends(get_db)):
    """Handle SAML callback from Identity Provider"""
    req = prepare_fastapi_request(request)
    auth = init_saml_auth(req)
    
    # Process SAML response
    auth.process_response()
    
    errors = auth.get_errors()
    if len(errors) == 0:
        # SAML authentication successful
        if not auth.is_authenticated():
            raise HTTPException(status_code=401, detail="SAML authentication failed")
        
        # Extract user information
        user_info = extract_user_info_from_saml(auth)
        
        # Validate university email
        if not is_valid_university_email(user_info["email"]):
            raise HTTPException(
                status_code=403, 
                detail=f"Access restricted to university accounts (@{', @'.join(settings.ALLOWED_EMAIL_DOMAINS)})"
            )
        
        # Determine user roles
        roles = determine_user_role(user_info)
        
        # Create or update user
        result = await db.execute(
            select(User).where(User.email == user_info["email"])
        )
        user = result.scalar_one_or_none()
        
        if user:
            # Update existing user
            user.saml_nameid = auth.get_nameid()
            user.last_login = datetime.utcnow()
            user.first_name = user_info["first_name"] or user.first_name
            user.last_name = user_info["last_name"] or user.last_name
            user.full_name = f"{user.first_name} {user.last_name}".strip() if user.first_name or user.last_name else user.full_name
            user.student_id = user_info["student_id"] or user.student_id
            user.affiliation = user_info["affiliation"] or user.affiliation
            
            # Update roles
            user.is_student = roles["is_student"]
            user.is_faculty = roles["is_faculty"]
            user.is_staff = roles["is_staff"]
            if roles["is_admin"]:  # Only elevate to admin, don't remove existing admin rights
                user.is_admin = True
        else:
            # Create new user
            user = User(
                email=user_info["email"],
                username=user_info["username"] or user_info["email"].split('@')[0],
                first_name=user_info["first_name"],
                last_name=user_info["last_name"],
                full_name=f"{user_info['first_name']} {user_info['last_name']}".strip(),
                student_id=user_info["student_id"],
                affiliation=user_info["affiliation"],
                saml_nameid=auth.get_nameid(),
                last_login=datetime.utcnow(),
                is_active=True,
                **roles
            )
            db.add(user)
        
        await db.commit()
        await db.refresh(user)
        
        # Create JWT token
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "username": user.username,
            "is_admin": user.is_admin
        }
        access_token = create_access_token(data=token_data)
        
        # Return success response with token
        return JSONResponse(
            content={
                "success": True,
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "username": user.username,
                    "name": user.full_name,
                    "is_admin": user.is_admin,
                    "is_student": user.is_student,
                    "is_faculty": user.is_faculty,
                    "is_staff": user.is_staff
                }
            }
        )
    else:
        # SAML authentication failed
        error_reason = auth.get_last_error_reason()
        raise HTTPException(
            status_code=401, 
            detail=f"SAML authentication failed: {error_reason}"
        )

@app.get("/auth/saml/logout", tags=["Authentication"])
async def saml_logout(request: Request, current_user: User = Depends(get_current_user)):
    """Initiate SAML logout"""
    req = prepare_fastapi_request(request)
    auth = init_saml_auth(req)
    
    # Get the NameID from the user's SAML session
    name_id = current_user.saml_nameid
    
    # Initiate logout
    slo_url = auth.logout(name_id=name_id)
    return RedirectResponse(url=slo_url, status_code=302)

@app.post("/auth/logout", tags=["Authentication"])
async def logout():
    """Simple logout (client should discard the token)"""
    return {"message": "Logged out successfully"}

# --- Admin Endpoints ---

@app.get("/admin/users", response_model=List[UserResponse], tags=["Admin"])
async def get_all_users(
    _: bool = Depends(require_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role_filter: Optional[str] = Query(None, regex="^(student|faculty|staff|admin)$"),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get all users with their prompt counts and filtering options"""
    query = select(User)
    
    # Apply role filter
    if role_filter:
        if role_filter == "student":
            query = query.where(User.is_student == True)
        elif role_filter == "faculty":
            query = query.where(User.is_faculty == True)
        elif role_filter == "staff":
            query = query.where(User.is_staff == True)
        elif role_filter == "admin":
            query = query.where(User.is_admin == True)
    
    # Apply search filter
    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            (User.email.ilike(search_pattern)) |
            (User.username.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern)) |
            (User.first_name.ilike(search_pattern)) |
            (User.last_name.ilike(search_pattern)) |
            (User.student_id.ilike(search_pattern))
        )
    
    query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
    users_result = await db.execute(query)
    users = users_result.scalars().all()
    
    response_users = []
    for user in users:
        # Count user prompts
        prompt_count_query = select(func.count(UserPrompt.id)).where(UserPrompt.user_id == user.id)
        prompt_count_result = await db.execute(prompt_count_query)
        prompt_count = prompt_count_result.scalar() or 0
        
        user_response = UserResponse.model_validate(user)
        user_response.prompt_count = prompt_count
        response_users.append(user_response)
    
    return response_users

@app.post("/admin/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Admin"])
async def create_user(
    user_data: UserCreate,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user"""
    # Check if user already exists
    existing_user = await db.execute(
        select(User).where(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    if existing_user.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email or username already exists"
        )
    
    new_user = User(**user_data.model_dump())
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    response = UserResponse.model_validate(new_user)
    response.prompt_count = 0
    return response

@app.put("/admin/users/{user_id}", response_model=UserResponse, tags=["Admin"])
async def update_user(
    user_id: uuid.UUID,
    user_data: UserUpdate,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update a user"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    for field, value in user_data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    
    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)
    
    response = UserResponse.model_validate(user)
    # Get prompt count
    prompt_count_query = select(func.count(UserPrompt.id)).where(UserPrompt.user_id == user.id)
    prompt_count_result = await db.execute(prompt_count_query)
    response.prompt_count = prompt_count_result.scalar() or 0
    
    return response

@app.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def delete_user(
    user_id: uuid.UUID,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a user and all their data"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.delete(user)
    await db.commit()

@app.get("/admin/users/{user_id}/prompts", response_model=List[UserPromptResponse], tags=["Admin"])
async def get_user_prompts(
    user_id: uuid.UUID,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all prompts for a specific user"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    prompts_query = select(UserPrompt).where(UserPrompt.user_id == user_id).order_by(UserPrompt.created_at.desc())
    prompts_result = await db.execute(prompts_query)
    prompts = prompts_result.scalars().all()
    
    return [UserPromptResponse.model_validate(prompt) for prompt in prompts]

@app.delete("/admin/users/{user_id}/prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def delete_user_prompt(
    user_id: uuid.UUID,
    prompt_id: uuid.UUID,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific user prompt"""
    prompt = await db.execute(
        select(UserPrompt).where(
            and_(UserPrompt.id == prompt_id, UserPrompt.user_id == user_id)
        )
    )
    prompt = prompt.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    await db.delete(prompt)
    await db.commit()

@app.get("/admin/library/submissions", response_model=List[LibrarySubmissionResponse], tags=["Admin"])
async def get_library_submissions(
    _: bool = Depends(require_admin),
    status_filter: Optional[str] = Query(None, regex="^(PENDING|APPROVED|REJECTED)$"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """Get library submission requests"""
    query = select(LibrarySubmission).options(selectinload(LibrarySubmission.user))
    
    if status_filter:
        query = query.where(LibrarySubmission.status == status_filter)
    
    query = query.offset(skip).limit(limit).order_by(LibrarySubmission.created_at.desc())
    result = await db.execute(query)
    submissions = result.scalars().all()
    
    response_submissions = []
    for submission in submissions:
        submission_response = LibrarySubmissionResponse.model_validate(submission)
        submission_response.user_email = submission.user.email if submission.user else None
        response_submissions.append(submission_response)
    
    return response_submissions

@app.post("/admin/library/submissions/{submission_id}/review", tags=["Admin"])
async def review_library_submission(
    submission_id: uuid.UUID,
    review_data: AdminReviewRequest,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve or reject a library submission"""
    submission = await db.get(LibrarySubmission, submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if submission.status != "PENDING":
        raise HTTPException(status_code=400, detail="Submission already reviewed")
    
    if review_data.action not in ["approve", "reject"]:
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")
    
    # Get a valid admin user ID from the database
    admin_user = await db.execute(
        select(User).where(User.is_admin == True).limit(1)
    )
    admin_user = admin_user.scalar_one_or_none()
    
    if not admin_user:
        # If no admin user exists, create one
        admin_user = User(
            email="admin@example.com",
            username="admin",
            full_name="System Administrator",
            is_active=True,
            is_admin=True
        )
        db.add(admin_user)
        await db.flush()  # Flush to get the ID
    
    # Update submission
    submission.status = "APPROVED" if review_data.action == "approve" else "REJECTED"
    submission.admin_notes = review_data.admin_notes
    submission.reviewed_at = datetime.utcnow()
    submission.reviewed_by = admin_user.id
    
    # If approved, create library prompt
    if review_data.action == "approve":
        new_library_prompt = LibraryPrompt(
            user_id=submission.user_id,
            submission_id=submission.id,
            text=submission.prompt_text,
            summary=submission.summary,
            tags=submission.tags
        )
        db.add(new_library_prompt)
    
    await db.commit()
    
    return {
        "success": True,
        "action": review_data.action,
        "submission_id": submission.id
    }

@app.delete("/admin/library/submissions/{submission_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def delete_library_submission(
    submission_id: uuid.UUID,
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a library submission"""
    submission = await db.get(LibrarySubmission, submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    await db.delete(submission)
    await db.commit()

@app.get("/admin/stats", tags=["Admin"])
async def get_admin_stats(
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get platform statistics"""
    # User stats
    total_users_result = await db.execute(select(func.count(User.id)))
    total_users = total_users_result.scalar() or 0
    
    active_users_result = await db.execute(select(func.count(User.id)).where(User.is_active == True))
    active_users = active_users_result.scalar() or 0
    
    admin_users_result = await db.execute(select(func.count(User.id)).where(User.is_admin == True))
    admin_users = admin_users_result.scalar() or 0
    
    # Prompt stats
    total_user_prompts_result = await db.execute(select(func.count(UserPrompt.id)))
    total_user_prompts = total_user_prompts_result.scalar() or 0
    
    total_library_prompts_result = await db.execute(select(func.count(LibraryPrompt.id)))
    total_library_prompts = total_library_prompts_result.scalar() or 0
    
    # Submission stats
    pending_submissions_result = await db.execute(
        select(func.count(LibrarySubmission.id)).where(LibrarySubmission.status == "PENDING")
    )
    pending_submissions = pending_submissions_result.scalar() or 0
    
    # Task stats
    total_tasks_result = await db.execute(select(func.count(Task.id)))
    total_tasks = total_tasks_result.scalar() or 0
    
    successful_tasks_result = await db.execute(select(func.count(Task.id)).where(Task.status == "SUCCESS"))
    successful_tasks = successful_tasks_result.scalar() or 0
    
    # Calculate success rate
    success_rate = round((successful_tasks / max(total_tasks, 1)) * 100, 2) if total_tasks > 0 else 0
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "admins": admin_users
        },
        "prompts": {
            "user_prompts": total_user_prompts,
            "library_prompts": total_library_prompts
        },
        "submissions": {
            "pending": pending_submissions
        },
        "tasks": {
            "total": total_tasks,
            "successful": successful_tasks,
            "success_rate": success_rate
        }
    }

# --- User Endpoints ---

@app.get("/user/prompts", response_model=List[UserPromptResponse], tags=["User"])
async def get_user_prompt_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's prompt history (top 20)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    prompts_query = select(UserPrompt).where(
        UserPrompt.user_id == current_user.id
    ).order_by(UserPrompt.created_at.desc()).limit(20)
    
    prompts_result = await db.execute(prompts_query)
    prompts = prompts_result.scalars().all()
    
    return [UserPromptResponse.model_validate(prompt) for prompt in prompts]

@app.post("/user/library/submit", response_model=dict, status_code=status.HTTP_201_CREATED, tags=["User"])
async def submit_to_library(
    request: LibrarySubmissionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit a prompt for library approval"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check for duplicate submissions
    existing = await db.execute(
        select(LibrarySubmission).where(
            and_(
                LibrarySubmission.prompt_text == request.prompt_text,
                LibrarySubmission.user_id == current_user.id,
                LibrarySubmission.status == "PENDING"
            )
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409, 
            detail="You have already submitted this prompt for review"
        )
    
    # Create submission
    submission = LibrarySubmission(
        user_id=current_user.id,
        prompt_text=request.prompt_text,
        submission_notes=request.submission_notes
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)
    
    # Trigger analysis task
    analyze_submission_task.delay(str(submission.id))
    
    return {
        "success": True,
        "submission_id": submission.id,
        "message": "Prompt submitted for admin review"
    }

@app.get("/user/library/submissions", response_model=List[LibrarySubmissionResponse], tags=["User"])
async def get_user_submissions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's library submissions"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    submissions_query = select(LibrarySubmission).where(
        LibrarySubmission.user_id == current_user.id
    ).order_by(LibrarySubmission.created_at.desc())
    
    submissions_result = await db.execute(submissions_query)
    submissions = submissions_result.scalars().all()
    
    return [LibrarySubmissionResponse.model_validate(sub) for sub in submissions]

# --- Library Endpoints (Updated) ---

@app.get("/library/prompts", response_model=List[LibraryPromptResponse], tags=["Library"])
async def get_library_prompts(
    user_id: Optional[str] = Query(None), 
    db: AsyncSession = Depends(get_db)
):
    """Get approved library prompts"""
    prompts = (await db.execute(
        select(LibraryPrompt).options(selectinload(LibraryPrompt.stars)).order_by(LibraryPrompt.created_at.desc())
    )).scalars().all()
    
    response_prompts = []
    for prompt in prompts:
        # Check if user_id is a valid UUID before comparing
        is_starred_by_user = False
        if user_id:
            try:
                # Try to parse as UUID to validate
                user_uuid = uuid.UUID(user_id)
                is_starred_by_user = any(star.user_id == user_uuid for star in prompt.stars)
            except ValueError:
                # Invalid UUID format, skip star check
                is_starred_by_user = False
        
        # Ensure prompt.user_id is a valid UUID, if not skip this prompt or handle gracefully
        try:
            prompt_user_id = prompt.user_id if isinstance(prompt.user_id, uuid.UUID) else uuid.UUID(str(prompt.user_id))
        except (ValueError, TypeError):
            # Skip prompts with invalid user_id
            continue
            
        response_prompts.append(
            LibraryPromptResponse(
                id=prompt.id, 
                user_id=prompt_user_id, 
                text=prompt.text,
                created_at=prompt.created_at, 
                summary=prompt.summary, 
                tags=prompt.tags,
                stars=len(prompt.stars),
                is_starred_by_user=is_starred_by_user
            )
        )
    
    response_prompts.sort(key=lambda p: p.stars, reverse=True)
    return response_prompts

@app.delete("/library/prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Library"])
async def delete_library_prompt(
    prompt_id: uuid.UUID, 
    _: bool = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a library prompt (admin only)"""
    prompt = await db.get(LibraryPrompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found.")
    await db.delete(prompt)
    await db.commit()

@app.post("/library/prompts/{prompt_id}/toggle-star", tags=["Library"])
async def toggle_star_for_prompt(
    prompt_id: uuid.UUID, 
    request: ToggleStarRequest, 
    db: AsyncSession = Depends(get_db)
):
    """Toggle star for a library prompt"""
    prompt = await db.get(LibraryPrompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found.")

    # Validate user_id is a valid UUID
    try:
        user_uuid = uuid.UUID(request.user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    stmt = select(PromptStar).where(
        PromptStar.prompt_id == prompt_id, 
        PromptStar.user_id == user_uuid
    )
    existing_star = (await db.execute(stmt)).scalar_one_or_none()

    if existing_star:
        await db.delete(existing_star)
        await db.commit()
        return {"success": True, "action": "unstarred"}
    else:
        new_star = PromptStar(prompt_id=prompt_id, user_id=user_uuid)
        db.add(new_star)
        await db.commit()
        return {"success": True, "action": "starred"}

# --- Existing Prompt Endpoints (Updated with User Tracking) ---

@app.post("/prompts/generate", response_model=TaskCreationResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Prompt Generation"])
async def generate_new_prompt(
    request: InitialGenerationRequest, 
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a new prompt"""
    new_task = Task(
        task_type="initial_generation",
        user_id=current_user.id if current_user else None
    )
    db.add(new_task)
    await db.commit()
    await db.refresh(new_task)
    create_initial_prompt_task.delay(str(new_task.id), request.model_dump())
    return TaskCreationResponse(task_id=new_task.id, status_url=f"/tasks/{new_task.id}")

@app.post("/prompts/evaluate", response_model=TaskCreationResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Prompt Evaluation"])
async def evaluate_existing_prompt(
    request: EvaluationRequest, 
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Evaluate an existing prompt"""
    new_task = Task(
        task_type="evaluation_and_improvement",
        user_id=current_user.id if current_user else None
    )
    db.add(new_task)
    await db.commit()
    await db.refresh(new_task)
    evaluate_prompt_task.delay(str(new_task.id), request.model_dump())
    return TaskCreationResponse(task_id=new_task.id, status_url=f"/tasks/{new_task.id}")

@app.post("/prompts/suggest-improvements", response_model=TaskCreationResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Prompt Optimization"])
async def get_improvement_suggestions(
    request: SuggestionRequest, 
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get improvement suggestions for a prompt"""
    new_task = Task(
        task_type="suggestion_generation",
        user_id=current_user.id if current_user else None
    )
    db.add(new_task)
    await db.commit()
    await db.refresh(new_task)
    generate_suggestions_task.delay(str(new_task.id), request.model_dump())
    return TaskCreationResponse(task_id=new_task.id, status_url=f"/tasks/{new_task.id}")

@app.post("/prompts/analyze-and-tag", response_model=TaskCreationResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Prompt Analysis"])
async def analyze_and_tag_prompt(
    request: AnalysisRequest, 
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze and tag a prompt"""
    new_task = Task(
        task_type="analysis_and_tagging",
        user_id=current_user.id if current_user else None
    )
    db.add(new_task)
    await db.commit()
    await db.refresh(new_task)
    analyze_and_tag_task.delay(str(new_task.id), request.model_dump())
    return TaskCreationResponse(task_id=new_task.id, status_url=f"/tasks/{new_task.id}")

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Task Management"])
async def get_task_status(task_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get task status and results"""
    stmt = select(Task).where(Task.id == task_id).options(
        selectinload(Task.generated_prompt),
        selectinload(Task.prompt_evaluation),
        selectinload(Task.prompt_suggestions),
        selectinload(Task.prompt_analysis),
        selectinload(Task.library_prompt)
    )
    task = (await db.execute(stmt)).scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response_data = {
        "task_id": task.id, 
        "task_type": task.task_type, 
        "status": task.status,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error_message": task.error_message, 
        "result": None
    }

    if task.status == "SUCCESS":
        if task.task_type == "initial_generation" and task.generated_prompt:
            response_data["result"] = GeneratedPromptResponse.model_validate(task.generated_prompt)
        elif task.task_type == "evaluation_and_improvement" and task.prompt_evaluation:
            response_data["result"] = EvaluationResponse.model_validate(task.prompt_evaluation)
        elif task.task_type == "suggestion_generation" and task.prompt_suggestions:
            response_data["result"] = [SuggestionResponse.model_validate(s) for s in task.prompt_suggestions]
        elif task.task_type == "analysis_and_tagging" and task.prompt_analysis:
            analysis = task.prompt_analysis
            response_data["result"] = AnalysisResponse(
                summary=analysis.summary,
                tags=analysis.tags or [],
                quality_indicators={
                    "clarity": analysis.clarity,
                    "bias_risk": analysis.bias_risk,
                    "safety_level": analysis.safety_level,
                    "completeness": analysis.completeness,
                    "professional_grade": analysis.professional_grade
                },
                category_analysis={
                    "primary_domain": analysis.primary_domain,
                    "main_purpose": analysis.main_purpose,
                    "target_audience": analysis.target_audience,
                    "complexity_level": analysis.complexity_level
                }
            )
        elif task.task_type == "add_to_library" and task.library_prompt:
            response_data["result"] = LibraryPromptResponse.model_validate(task.library_prompt)

    return TaskStatusResponse(**response_data)