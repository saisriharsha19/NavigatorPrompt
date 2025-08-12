
'use server';

import type { GenerateInitialPromptOutput } from '@/ai/flows/generate-initial-prompt';
import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import type { GeneratePromptSuggestionsOutput } from '@/ai/flows/get-prompt-suggestions';
import { db } from '@/lib/db';
import type { Prompt, LibrarySubmission, User, PlatformStats } from '@/hooks/use-prompts';
import { revalidatePath } from 'next/cache';
import type { GeneratePromptMetadataOutput } from '@/ai/flows/generate-prompt-tags';

const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5000';
const ADMIN_KEY = process.env.ADMIN_KEY;

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

// Mock token for demonstration purposes. In a real app, this would be a real JWT.
const MOCK_AUTH_TOKEN = "mock-user-123-token";

// --- Task-based API Calls ---

export type TaskCreationResponse = {
  task_id: string;
  status_url: string;
};

export async function handleGenerateInitialPrompt(input: { user_needs: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/generate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
      },
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
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
       },
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
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
      },
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
    const response = await fetch(`${BACKEND_URL}${status_url}`, {
        headers: { 'Authorization': `Bearer ${MOCK_AUTH_TOKEN}` },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Failed to get task status from ${status_url}.`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching task result from ${status_url}:`, error);
    return {
      task_id: status_url.split('/').pop() || 'unknown',
      task_type: 'unknown',
      status: 'FAILURE',
      created_at: new Date().toISOString(),
      error_message: getErrorMessage(error),
    };
  }
}


// --- Database Actions ---

export async function getHistoryPromptsFromDB(userId: string): Promise<Prompt[]> {
  if (!userId) return [];
  try {
    const response = await fetch(`${BACKEND_URL}/user/prompts`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
        },
        cache: 'no-store',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to fetch user history.');
    }

    const prompts: Prompt[] = await response.json();
    return prompts;
  } catch (error) {
    console.error('Failed to get history prompts:', error);
    return [];
  }
}

// History is now read-only for users. Deletion is an admin action.
// The ADMIN_KEY should be stored securely and not exposed on the client-side.
// This function assumes an admin context if an admin key is available.
export async function deleteHistoryPromptFromDB(id: string, userId: string): Promise<{ success: boolean }> {
  if (!ADMIN_KEY) {
      throw new Error("Admin action required, but no admin key is configured.");
  }
  if (!userId) throw new Error('User ID is required to delete a prompt.');

  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}/prompts/${id}`, {
        method: 'DELETE',
        headers: {
            'X-Admin-Key': ADMIN_KEY
        },
    });

    if (response.status === 204) {
      revalidatePath('/history');
      revalidatePath(`/admin/users/${userId}`);
      return { success: true };
    }
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    revalidatePath('/history');
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
      cache: 'no-store',
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

export async function submitPromptToLibrary(request: { prompt_text: string, submission_notes?: string }): Promise<LibrarySubmission> {
    try {
      const response = await fetch(`${BACKEND_URL}/user/library/submit`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
          },
          body: JSON.stringify(request),
      });
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
      }
      
      const newSubmission: LibrarySubmission = await response.json();
      revalidatePath('/library');
      return newSubmission;

    } catch (error: any) {
      console.error('Failed to submit prompt to library via API:', error);
      throw new Error(error.message || 'Failed to submit prompt to library.');
    }
}


export async function deleteLibraryPromptFromDB(promptId: string): Promise<{ success: boolean }> {
  if (!ADMIN_KEY) {
      throw new Error("Admin action required, but no admin key is configured.");
  }

  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}`, {
        method: 'DELETE',
        headers: {
            'X-Admin-Key': ADMIN_KEY
        },
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
            'Authorization': `Bearer ${MOCK_AUTH_TOKEN}`,
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

// --- Admin Actions ---

export async function getAdminStats(): Promise<PlatformStats> {
  if (!ADMIN_KEY) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/stats`, {
      headers: { 'X-Admin-Key': ADMIN_KEY },
      cache: 'no-store',
    });
    if (!response.ok) throw new Error('Failed to fetch admin stats.');
    return await response.json();
  } catch (error) {
    console.error('Error fetching admin stats:', error);
    // Return a default/empty stats object on error
    return {
      total_users: 0,
      total_prompts_in_history: 0,
      total_prompts_in_library: 0,
      pending_submissions: 0,
    };
  }
}

export async function getAdminUsers(): Promise<User[]> {
  if (!ADMIN_KEY) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users`, {
      headers: { 'X-Admin-Key': ADMIN_KEY },
      cache: 'no-store',
    });
    if (!response.ok) throw new Error('Failed to fetch users.');
    return await response.json();
  } catch (error) {
    console.error('Error fetching users:', error);
    return [];
  }
}

export async function getAdminLibrarySubmissions(status: 'PENDING' | 'APPROVED' | 'REJECTED'): Promise<LibrarySubmission[]> {
  if (!ADMIN_KEY) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions?status=${status}`, {
      headers: { 'X-Admin-Key': ADMIN_KEY },
      cache: 'no-store',
    });
    if (!response.ok) throw new Error('Failed to fetch library submissions.');
    return await response.json();
  } catch (error) {
    console.error('Error fetching library submissions:', error);
    return [];
  }
}

export async function reviewLibrarySubmission(submissionId: string, action: 'approve' | 'reject', adminNotes?: string): Promise<{ success: boolean }> {
  if (!ADMIN_KEY) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions/${submissionId}/review`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Admin-Key': ADMIN_KEY,
      },
      body: JSON.stringify({ action, admin_notes: adminNotes }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to review submission.');
    }
    revalidatePath('/admin/submissions');
    return { success: true };
  } catch (error) {
    console.error('Error reviewing submission:', error);
    throw error;
  }
}
