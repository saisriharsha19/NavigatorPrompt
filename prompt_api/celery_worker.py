# celery_worker.py
import asyncio
import json
import httpx
import sys
import re # Import the regular expression module
from celery import Celery
from celery.utils.log import get_task_logger
from datetime import datetime
import nest_asyncio

# Apply nest_asyncio patch ONLY if running as a Celery worker.
if "celery" in sys.argv[0]:
    nest_asyncio.apply()

from config import settings
from prompts import PROMPT_TEMPLATES
from database import AsyncSessionLocal
from models import (
    Task, GeneratedPrompt, PromptEvaluation, PromptSuggestion, PromptAnalysis, 
    LibraryPrompt, LibrarySubmission, UserPrompt, User
)
from sqlalchemy.future import select
from sqlalchemy import desc

logger = get_task_logger(__name__)

# Configure Celery with Sentinel for High Availability
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    broker_transport_options={
        'master_name': settings.REDIS_SERVICE_NAME,
    },
    result_backend_transport_options={
        'master_name': settings.REDIS_SERVICE_NAME,
    }
)
celery_app.conf.update(task_track_started=True)


def extract_json_from_response(llm_content: str) -> str:
    """
    Robustly extracts a JSON object from a string.
    """
    # Try to find JSON within ```json ... ``` fences
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_content, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Fall back to finding the first and last curly brace
    json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return llm_content


async def manage_user_prompt_history(user_id: str, prompt_text: str, task_type: str, session):
    """
    Manages user's prompt history, keeping only top 20 most recent prompts.
    """
    try:
        # Add new prompt to history
        user_prompt = UserPrompt(
            user_id=user_id,
            prompt_text=prompt_text,
            task_type=task_type
        )
        session.add(user_prompt)
        
        # Get current count of user prompts
        count_result = await session.execute(
            select(UserPrompt).where(UserPrompt.user_id == user_id)
        )
        user_prompts = count_result.scalars().all()
        
        # If more than 20, delete oldest ones
        if len(user_prompts) >= 20:
            # Get oldest prompts to delete
            oldest_prompts = await session.execute(
                select(UserPrompt)
                .where(UserPrompt.user_id == user_id)
                .order_by(UserPrompt.created_at.asc())
                .limit(len(user_prompts) - 19)  # Keep 19 + new one = 20
            )
            
            for old_prompt in oldest_prompts.scalars():
                await session.delete(old_prompt)
                
        await session.commit()
        logger.info(f"Updated prompt history for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error managing user prompt history: {e}")
        await session.rollback()


# --- Celery Tasks ---

@celery_app.task(name="tasks.analyze_submission")
def analyze_submission_task(submission_id: str):
    """
    Analyzes a library submission and saves the analysis results.
    """
    logger.info(f"Executing analyze_submission_task for submission_id: {submission_id}")
    
    async def run_task():
        session = AsyncSessionLocal()
        try:
            submission = await session.get(LibrarySubmission, submission_id)
            if not submission:
                logger.error(f"Submission {submission_id} not found.")
                return

            # Create a task for this analysis
            task = Task(task_type="submission_analysis", user_id=submission.user_id)
            session.add(task)
            await session.commit()
            await session.refresh(task)
            
            # Update submission with task_id
            submission.task_id = task.id
            
            # Analyze the prompt
            analysis_template = PROMPT_TEMPLATES["analysis_and_tagging"]
            formatted_prompt = analysis_template["template"].format(
                promptText=submission.prompt_text, targetedContext=""
            )
            
            headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
            payload = {"model": settings.MODEL_NAME, "messages": [{"role": "user", "content": formatted_prompt}]}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(settings.BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
            llm_content = response.json()["choices"][0]["message"]["content"]
            cleaned_content = extract_json_from_response(llm_content)
            result_json = json.loads(cleaned_content)

            quality = result_json.get("qualityIndicators", {})
            category = result_json.get("categoryAnalysis", {})
            raw_pg = quality.get("professional_grade", False)
            professional_grade_bool = str(raw_pg).lower() in ['true', '1']

            new_analysis = PromptAnalysis(
                task_id=task_id, summary=result_json.get("summary"), tags=result_json.get("tags"),
                clarity=quality.get("clarity"), bias_risk=quality.get("bias_risk"),
                safety_level=quality.get("safety_level"), completeness=quality.get("completeness"),
                professional_grade=professional_grade_bool,
                primary_domain=category.get("primary_domain"), main_purpose=category.get("main_purpose"),
                target_audience=category.get("target_audience"), complexity_level=category.get("complexity_level")
            )
            session.add(new_analysis)
            
            # Manage user prompt history
            if task.user_id:
                await manage_user_prompt_history(
                    str(task.user_id), 
                    request_data.get("prompt_text", ""), 
                    "analysis", 
                    session
                )
            
            task.status = "SUCCESS"
            task.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}", exc_info=True)
            await session.rollback()
            async with AsyncSessionLocal() as error_session:
                 task_to_update = await error_session.get(Task, task_id)
                 if task_to_update:
                    task_to_update.status = "FAILURE"
                    task_to_update.error_message = str(e)
                    task_to_update.completed_at = datetime.utcnow()
                    await error_session.commit()
        finally:
            await session.close()
    asyncio.run(run_task())

@celery_app.task(name="tasks.create_initial_prompt")
def create_initial_prompt_task(task_id: str, request_data: dict):
    logger.info(f"Executing create_initial_prompt_task for task_id: {task_id}")
    async def run_task():
        session = AsyncSessionLocal()
        try:
            task = await session.get(Task, task_id)
            if not task: return

            template_data = PROMPT_TEMPLATES["initial_generation"]
            formatted_prompt = template_data["template"].format(
                userNeeds=request_data.get("user_needs", ""),
                deepevalContext=request_data.get("deepeval_context", "")
            )
            headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
            payload = {"model": settings.MODEL_NAME, "messages": [{"role": "user", "content": formatted_prompt}]}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(settings.BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
            llm_content = response.json()["choices"][0]["message"]["content"]
            
            initial_prompt_text = llm_content
            try:
                cleaned_content = extract_json_from_response(llm_content)
                result_json = json.loads(cleaned_content)
                initial_prompt_text = result_json.get("initialPrompt", llm_content)
            except json.JSONDecodeError:
                logger.warning(f"Task {task_id}: LLM did not return valid JSON. Using raw content.")

            new_prompt = GeneratedPrompt(
                task_id=task_id, user_needs=request_data["user_needs"],
                initial_prompt=initial_prompt_text
            )
            session.add(new_prompt)
            
            # Manage user prompt history
            if task.user_id:
                await manage_user_prompt_history(
                    str(task.user_id), 
                    initial_prompt_text, 
                    "generation", 
                    session
                )
            
            task.status = "SUCCESS"
            task.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}", exc_info=True)
            await session.rollback()
            async with AsyncSessionLocal() as error_session:
                 task_to_update = await error_session.get(Task, task_id)
                 if task_to_update:
                    task_to_update.status = "FAILURE"
                    task_to_update.error_message = str(e)
                    task_to_update.completed_at = datetime.utcnow()
                    await error_session.commit()
        finally:
            await session.close()
    asyncio.run(run_task())

@celery_app.task(name="tasks.evaluate_prompt")
def evaluate_prompt_task(task_id: str, request_data: dict):
    logger.info(f"Executing evaluate_prompt_task for task_id: {task_id}")
    async def run_task():
        session = AsyncSessionLocal()
        try:
            task = await session.get(Task, task_id)
            if not task: return

            template_data = PROMPT_TEMPLATES["evaluation_and_improvement"]
            formatted_prompt = template_data["template"].format(
                prompt=request_data.get("prompt", ""), userNeeds=request_data.get("user_needs", ""),
                retrievedContentSection=f"**KNOWLEDGE BASE CONTENT:**\n{request_data.get('retrieved_content', '')}" if request_data.get("retrieved_content") else "",
                groundTruthsSection=f"**GROUND TRUTH EXAMPLES:**\n{request_data.get('ground_truths', '')}" if request_data.get("ground_truths") else "",
                deepevalContext="", faithfulnessSection=""
            )
            headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
            payload = {"model": settings.MODEL_NAME, "messages": [{"role": "user", "content": formatted_prompt}]}
            async with httpx.AsyncClient(timeout=400.0) as client:
                response = await client.post(settings.BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
            llm_content = response.json()["choices"][0]["message"]["content"]
            cleaned_content = extract_json_from_response(llm_content)
            result_json = json.loads(cleaned_content)
            
            bias_data = result_json.get("bias", {})
            toxicity_data = result_json.get("toxicity", {})
            alignment_data = result_json.get("promptAlignment", {})
            
            new_eval = PromptEvaluation(
                task_id=task_id, original_prompt=request_data["prompt"],
                improved_prompt=result_json.get("improvedPrompt"), improvement_summary=result_json.get("improvementSummary"),
                bias_score=bias_data.get("score"), bias_summary=bias_data.get("summary"),
                bias_issues=bias_data.get("issues"), bias_mitigations=bias_data.get("mitigations"),
                bias_test_cases=bias_data.get("testCases"),
                toxicity_score=toxicity_data.get("score"), toxicity_summary=toxicity_data.get("summary"),
                toxicity_risks=toxicity_data.get("risks"), toxicity_safeguards=toxicity_data.get("safeguards"),
                toxicity_test_cases=toxicity_data.get("testCases"),
                alignment_score=alignment_data.get("score"), alignment_summary=alignment_data.get("summary"),
                alignment_strengths=alignment_data.get("strengths"), alignment_improvements=alignment_data.get("improvements"),
                alignment_test_cases=alignment_data.get("testCases"),
            )
            session.add(new_eval)
            
            # Manage user prompt history
            if task.user_id:
                await manage_user_prompt_history(
                    str(task.user_id), 
                    request_data["prompt"], 
                    "evaluation", 
                    session
                )
            
            task.status = "SUCCESS"
            task.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}", exc_info=True)
            await session.rollback()
            async with AsyncSessionLocal() as error_session:
                 task_to_update = await error_session.get(Task, task_id)
                 if task_to_update:
                    task_to_update.status = "FAILURE"
                    task_to_update.error_message = str(e)
                    task_to_update.completed_at = datetime.utcnow()
                    await error_session.commit()
        finally:
            await session.close()
    asyncio.run(run_task())

@celery_app.task(name="tasks.generate_suggestions")
def generate_suggestions_task(task_id: str, request_data: dict):
    logger.info(f"Executing generate_suggestions_task for task_id: {task_id}")
    async def run_task():
        session = AsyncSessionLocal()
        try:
            task = await session.get(Task, task_id)
            if not task: return

            template_data = PROMPT_TEMPLATES["suggestion_generation"]
            formatted_prompt = template_data["template"].format(
                currentPrompt=request_data.get("current_prompt", ""),
                userCommentsSection=f"**USER FEEDBACK:**\n{request_data.get('user_comments', '')}" if request_data.get("user_comments") else "",
                targetedContext=f"**TARGETED CONTEXT:**\n{request_data.get('targeted_context', '')}" if request_data.get("targeted_context") else ""
            )
            headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
            payload = {"model": settings.MODEL_NAME, "messages": [{"role": "user", "content": formatted_prompt}]}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(settings.BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
            llm_content = response.json()["choices"][0]["message"]["content"]
            cleaned_content = extract_json_from_response(llm_content)
            result_json = json.loads(cleaned_content)
            suggestions = result_json.get("suggestions", [])
            
            for sug_data in suggestions:
                priority_score_val = sug_data.get("priority_score", 0)
                try:
                    priority_score_int = int(float(priority_score_val))
                except (ValueError, TypeError):
                    priority_score_int = 0
                new_suggestion = PromptSuggestion(
                    task_id=task_id, category=sug_data.get("category"), title=sug_data.get("title"),
                    description=sug_data.get("description"), implementation=sug_data.get("implementation"),
                    impact=sug_data.get("impact"), priority_score=priority_score_int
                )
                session.add(new_suggestion)
            
            # Manage user prompt history
            if task.user_id:
                await manage_user_prompt_history(
                    str(task.user_id), 
                    request_data.get("current_prompt", ""), 
                    "suggestions", 
                    session
                )
            
            task.status = "SUCCESS"
            task.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}", exc_info=True)
            await session.rollback()
            async with AsyncSessionLocal() as error_session:
                 task_to_update = await error_session.get(Task, task_id)
                 if task_to_update:
                    task_to_update.status = "FAILURE"
                    task_to_update.error_message = str(e)
                    task_to_update.completed_at = datetime.utcnow()
                    await error_session.commit()
        finally:
            await session.close()
    asyncio.run(run_task())

@celery_app.task(name="tasks.analyze_and_tag")
def analyze_and_tag_task(task_id: str, request_data: dict):
    logger.info(f"Executing analyze_and_tag_task for task_id: {task_id}")
    async def run_task():
        session = AsyncSessionLocal()
        try:
            task = await session.get(Task, task_id)
            if not task: return

            template_data = PROMPT_TEMPLATES["analysis_and_tagging"]
            formatted_prompt = template_data["template"].format(
                promptText=request_data.get("prompt_text", ""),
                targetedContext=f"**TARGETED CONTEXT:**\n{request_data.get('targeted_context', '')}" if request_data.get("targeted_context") else ""
            )
            headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
            payload = {"model": settings.MODEL_NAME, "messages": [{"role": "user", "content": formatted_prompt}]}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(settings.BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
            llm_content = response.json()["choices"][0]["message"]["content"]
            cleaned_content = extract_json_from_response(llm_content)
            result_json = json.loads(cleaned_content)

            quality = result_json.get("qualityIndicators", {})
            category = result_json.get("categoryAnalysis", {})
            raw_pg = quality.get("professional_grade", False)
            professional_grade_bool = str(raw_pg).lower() in ['true', '1']

            new_analysis = PromptAnalysis(
                task_id=task_id, summary=result_json.get("summary"), tags=result_json.get("tags"),
                clarity=quality.get("clarity"), bias_risk=quality.get("bias_risk"),
                safety_level=quality.get("safety_level"), completeness=quality.get("completeness"),
                professional_grade=professional_grade_bool,
                primary_domain=category.get("primary_domain"), main_purpose=category.get("main_purpose"),
                target_audience=category.get("target_audience"), complexity_level=category.get("complexity_level")
            )
            session.add(new_analysis)
            
            # Manage user prompt history
            if task.user_id:
                await manage_user_prompt_history(
                    str(task.user_id), 
                    request_data.get("prompt_text", ""), 
                    "analysis", 
                    session
                )
            
            task.status = "SUCCESS"
            task.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}", exc_info=True)
            await session.rollback()
            async with AsyncSessionLocal() as error_session:
                task_to_update = await error_session.get(Task, task_id)
                if task_to_update:
                    task_to_update.status = "FAILURE"
                    task_to_update.error_message = str(e)
                    task_to_update.completed_at = datetime.utcnow()
            await error_session.commit()
        finally:
            await session.close()
    asyncio.run(run_task())