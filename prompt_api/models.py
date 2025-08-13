# models.py
import uuid
from sqlalchemy import (
    Column, String, DateTime, JSON, ForeignKey, Text, Float, Integer, ARRAY, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

# User Management
class User(Base):
    """
    User management table for authentication and authorization.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    
    # University-specific fields
    student_id = Column(String, nullable=True, index=True)
    affiliation = Column(String, nullable=True)  # student, faculty, staff
    
    # Role flags
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    is_student = Column(Boolean, default=False)
    is_faculty = Column(Boolean, default=False)
    is_staff = Column(Boolean, default=False)
    
    # SAML authentication fields
    saml_nameid = Column(String, nullable=True, unique=True)
    last_login = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user_prompts = relationship("UserPrompt", back_populates="user", cascade="all, delete-orphan")
    library_submissions = relationship("LibrarySubmission", foreign_keys="LibrarySubmission.user_id", back_populates="user", cascade="all, delete-orphan")
    stars = relationship("PromptStar", back_populates="user", cascade="all, delete-orphan")

# User's Personal Prompt History (Top 20)
class UserPrompt(Base):
    """
    Stores user's personal prompt history with a limit of 20 prompts per user.
    """
    __tablename__ = "user_prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    task_type = Column(String, nullable=False)  # generation, evaluation, etc.
    created_at = Column(DateTime, server_default=func.now())
    
    # Optional metadata
    summary = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)

    user = relationship("User", back_populates="user_prompts")

# Library Submission Requests (Admin Approval Required)
class LibrarySubmission(Base):
    """
    Tracks requests to add prompts to the public library.
    Requires admin approval before becoming a LibraryPrompt.
    """
    __tablename__ = "library_submissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    submission_notes = Column(Text, nullable=True)  # User's notes about the prompt
    
    # Admin review fields
    status = Column(String, default="PENDING", index=True)  # PENDING, APPROVED, REJECTED
    admin_notes = Column(Text, nullable=True)
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Analysis results (filled by background task)
    summary = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    
    # Link to the task that analyzed this submission
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True, unique=True)

    user = relationship("User", foreign_keys=[user_id], back_populates="library_submissions")
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    task = relationship("Task", foreign_keys=[task_id])

# Central Task Tracking Table
class Task(Base):
    """
    A central table to track the status of all asynchronous Celery tasks.
    Each task, regardless of its type (generation, evaluation, etc.),
    will have an entry here. Specific results are stored in related tables.
    """
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)  # Track which user initiated
    task_type = Column(String, nullable=False, index=True) # e.g., 'initial_generation', 'evaluation'
    status = Column(String, default="PENDING", index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    # Relationships to specific result tables
    generated_prompt = relationship("GeneratedPrompt", back_populates="task", uselist=False)
    prompt_evaluation = relationship("PromptEvaluation", back_populates="task", uselist=False)
    prompt_suggestions = relationship("PromptSuggestion", back_populates="task")
    prompt_analysis = relationship("PromptAnalysis", back_populates="task", uselist=False)
    library_prompt = relationship("LibraryPrompt", back_populates="task", uselist=False)

# Updated Library Prompt (Admin Approved)
class LibraryPrompt(Base):
    """
    Stores prompts that have been analyzed and approved by admin for the community library.
    """
    __tablename__ = "library_prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("library_submissions.id"), nullable=True, unique=True)
    text = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, server_default=func.now())
    summary = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    
    # Link to the creation task
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True, unique=True)
    task = relationship("Task", back_populates="library_prompt")
    
    # Relationship to the stars, ensuring they are deleted if the prompt is deleted
    stars = relationship("PromptStar", back_populates="prompt", cascade="all, delete-orphan")
    submission = relationship("LibrarySubmission", foreign_keys=[submission_id])

class PromptStar(Base):
    """
    Tracks which user has starred which library prompt (many-to-many relationship).
    """
    __tablename__ = "prompt_stars"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("library_prompts.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    prompt = relationship("LibraryPrompt", back_populates="stars")
    user = relationship("User", back_populates="stars")


# Table for 'initial_generation' template results
class GeneratedPrompt(Base):
    """
    Stores the output from the 'initial_generation' template.
    """
    __tablename__ = "generated_prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False, unique=True)
    user_needs = Column(Text, nullable=False)
    initial_prompt = Column(Text, nullable=False)

    task = relationship("Task", back_populates="generated_prompt")


# Table for 'evaluation_and_improvement' template results
class PromptEvaluation(Base):
    """
    Stores the detailed evaluation metrics from the 'evaluation_and_improvement' template.
    """
    __tablename__ = "prompt_evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False, unique=True)
    
    original_prompt = Column(Text)
    improved_prompt = Column(Text)
    improvement_summary = Column(Text)

    # Bias Assessment
    bias_score = Column(Float)
    bias_summary = Column(Text)
    bias_issues = Column(ARRAY(String))
    bias_mitigations = Column(ARRAY(String))
    bias_test_cases = Column(ARRAY(String))

    # Toxicity Assessment
    toxicity_score = Column(Float)
    toxicity_summary = Column(Text)
    toxicity_risks = Column(ARRAY(String))
    toxicity_safeguards = Column(ARRAY(String))
    toxicity_test_cases = Column(ARRAY(String))

    # Prompt Alignment
    alignment_score = Column(Float)
    alignment_summary = Column(Text)
    alignment_strengths = Column(ARRAY(String))
    alignment_improvements = Column(ARRAY(String))
    alignment_test_cases = Column(ARRAY(String))

    task = relationship("Task", back_populates="prompt_evaluation")


# Table for 'suggestion_generation' template results
class PromptSuggestion(Base):
    """
    Stores an individual suggestion from the 'suggestion_generation' template.
    A single task can generate multiple suggestions.
    """
    __tablename__ = "prompt_suggestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)

    category = Column(String) # critical|high|medium|enhancement
    title = Column(String)
    description = Column(Text)
    implementation = Column(Text)
    impact = Column(Text)
    priority_score = Column(Integer)

    task = relationship("Task", back_populates="prompt_suggestions")


# Table for 'analysis_and_tagging' template results
class PromptAnalysis(Base):
    """
    Stores the analysis and tags from the 'analysis_and_tagging' template.
    """
    __tablename__ = "prompt_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False, unique=True)

    summary = Column(String(255))
    tags = Column(ARRAY(String))
    
    # Quality Indicators
    clarity = Column(String) # high|medium|low
    bias_risk = Column(String) # low|medium|high
    safety_level = Column(String) # high|medium|low
    completeness = Column(String) # high|medium|low
    professional_grade = Column(Boolean)
    
    # Category Analysis
    primary_domain = Column(String)
    main_purpose = Column(String)
    target_audience = Column(String)
    complexity_level = Column(String)
    
    task = relationship("Task", back_populates="prompt_analysis")