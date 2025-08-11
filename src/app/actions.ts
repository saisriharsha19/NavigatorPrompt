
'use server';

import type { GenerateInitialPromptOutput } from '@/ai/flows/generate-initial-prompt';
import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import type { GeneratePromptSuggestionsOutput } from '@/ai/flows/get-prompt-suggestions';
import { db } from '@/lib/db';
import type { Prompt } from '@/hooks/use-prompts';
import { revalidatePath } from 'next/cache';
import type { GeneratePromptMetadataOutput } from '@/ai/flows/generate-prompt-tags';

const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

// --- Task-based API Calls ---

export type TaskCreationResponse = {
  task_id: string;
  status_url: string;
};

export async function handleGenerateInitialPrompt(input: { user_needs: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start generation task.');
    }
    return await response.json();
  } catch (error) {
    console.error('Error in handleGenerateInitialPrompt:', error);
    throw new Error(`An error occurred while starting the generation task: ${getErrorMessage(error)}`);
  }
}

export async function handleEvaluatePrompt(input: { prompt: string; user_needs: string; }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start evaluation task.');
    }
    return await response.json();
  } catch (error) {
    console.error('Error in handleEvaluatePrompt:', error);
    throw new Error(`An error occurred while starting the evaluation task: ${getErrorMessage(error)}`);
  }
}

export async function handleGetPromptSuggestions(input: { current_prompt: string; user_comments?: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/suggest-improvements`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start suggestion task.');
    }
    return await response.json();
  } catch (error) {
    console.error('Error in handleGetPromptSuggestions:', error);
    throw new Error(`An error occurred while starting the suggestion task: ${getErrorMessage(error)}`);
  }
}

// This function is kept for conceptual mapping, but the backend doesn't have a direct "iterate" endpoint.
// Iteration is now: get suggestions -> user selects -> client-side prompt update -> re-evaluate.
// We'll return a dummy successful task for now to avoid breaking the UI flow.
export async function handleIterateOnPrompt(input: { currentPrompt: string; userComments: string; }): Promise<{ newPrompt: string }> {
  console.warn("handleIterateOnPrompt is a client-side operation now and should be refactored.");
  // This would be handled client-side by applying suggestions.
  // We return a simple object to satisfy the call.
  return { newPrompt: input.currentPrompt };
}

export type TaskStatusResponse = {
  task_id: string;
  task_type: string;
  status: 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE' | 'RETRY';
  created_at: string;
  completed_at?: string;
  error_message?: string;
  result?: GenerateInitialPromptOutput | EvaluateAndIteratePromptOutput | GeneratePromptSuggestionsOutput | GeneratePromptMetadataOutput;
};

export async function getTaskResult(status_url: string): Promise<TaskStatusResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}${status_url}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Failed to get task status from ${status_url}.`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching task result from ${status_url}:`, error);
    // Return a synthetic error response to be handled by the client
    return {
      task_id: status_url.split('/').pop() || 'unknown',
      task_type: 'unknown',
      status: 'FAILURE',
      created_at: new Date().toISOString(),
      error_message: getErrorMessage(error),
    };
  }
}


// --- Database Actions (History Only) ---
const MAX_HISTORY_PROMPTS_PER_USER = 20;

export async function getHistoryPromptsFromDB(userId: string): Promise<Prompt[]> {
  if (!userId) return [];
  try {
    const sql = 'SELECT * FROM prompts WHERE userId = ? ORDER BY createdAt DESC';
    const prompts = await db.query<Prompt>(sql, [userId]);
    return prompts;
  } catch (error) {
    console.error('Failed to get history prompts:', error);
    return [];
  }
}

export async function addHistoryPromptToDB(promptText: string, userId: string): Promise<Prompt> {
  if (!userId) throw new Error('User not authenticated.');

  try {
    const countSql = 'SELECT COUNT(*) as count FROM prompts WHERE userId = ?';
    const countResult = await db.query<{ count: number | string }>(countSql, [userId]);
    const count = Number(countResult[0]?.count || 0);

    if (count >= MAX_HISTORY_PROMPTS_PER_USER) {
      const oldestSql = 'DELETE FROM prompts WHERE id = (SELECT id FROM prompts WHERE userId = ? ORDER BY createdAt ASC LIMIT 1)';
      await db.run(oldestSql, [userId]);
    }

    const newPrompt: Prompt = {
      id: crypto.randomUUID(),
      userId: userId,
      text: promptText,
      createdAt: new Date().toISOString(),
    };

    const insertSql = 'INSERT INTO prompts (id, userId, text, createdAt) VALUES (?, ?, ?, ?)';
    await db.run(insertSql, [newPrompt.id, newPrompt.userId, newPrompt.text, newPrompt.createdAt]);
    revalidatePath('/history');
    return newPrompt;
  } catch (error: any) {
    console.error('Failed to add prompt to history:', error);
    throw new Error(error.message || 'Failed to save prompt to history.');
  }
}

export async function deleteHistoryPromptFromDB(id: string, userId: string): Promise<{ success: boolean }> {
  if (!userId) throw new Error('User not authenticated.');

  try {
    const sql = 'DELETE FROM prompts WHERE id = ? AND userId = ?';
    const result = await db.run(sql, [id, userId]);
    revalidatePath('/history');
    if (result.rowCount === 0) {
      throw new Error("Prompt not found or you don't have permission to delete it.");
    }
    return { success: true };
  } catch (error: any) {
    console.error('Failed to delete prompt from history:', error);
    throw new Error(error.message || 'Failed to delete prompt from history.');
  }
}


// --- Library Actions (API only) ---
export async function getLibraryPromptsFromDB(userId: string | null): Promise<Prompt[]> {
  try {
    const url = new URL(`${BACKEND_URL}/library/prompts`);
    if (userId) {
      url.searchParams.append('user_id', userId);
    }
    
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store', // Ensure fresh data
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    const prompts: Prompt[] = await response.json();
    return prompts;
  } catch (error) {
    console.error(`API call for library prompts failed: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function addLibraryPromptToDB(request: { prompt_text: string, user_id: string }): Promise<Prompt> {
    if (!request.user_id) throw new Error('User not authenticated.');
  
    try {
      const response = await fetch(`${BACKEND_URL}/library/prompts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
      });
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
      }
      
      const newPrompt: Prompt = await response.json();
      revalidatePath('/library');
      return newPrompt;

    } catch (error: any) {
      console.error('Failed to add prompt to library via API:', error);
      throw new Error(error.message || 'Failed to save prompt to library.');
    }
}

export async function deleteLibraryPromptFromDB(promptId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}`, {
        method: 'DELETE',
    });

    if (response.status === 204) {
      revalidatePath('/library');
      return { success: true };
    }
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    revalidatePath('/library');
    return { success: true };
  } catch (error: any) {
    console.error('Failed to delete prompt from library via API:', error);
    throw new Error(error.message || 'Failed to delete prompt from library.');
  }
}

export async function toggleStarForPrompt(promptId: string, request: { user_id: string }): Promise<{ success: boolean, action: 'starred' | 'unstarred' }> {
  if (!request.user_id) throw new Error('User not authenticated.');

  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}/toggle-star`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    revalidatePath('/library');
    const result = await response.json();

    return { 
      success: true,
      action: result.action,
    };

  } catch (error: any) {
     console.error('Failed to toggle star for prompt via API:', error);
    throw new Error(error.message || 'Failed to toggle star for prompt.');
  }
}
