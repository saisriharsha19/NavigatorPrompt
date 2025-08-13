
'use server';

import type { GenerateInitialPromptOutput } from '@/ai/flows/generate-initial-prompt';
import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import type { GeneratePromptSuggestionsOutput } from '@/ai/flows/get-prompt-suggestions';
import { db } from '@/lib/db';
import type { Prompt, LibrarySubmission, User, PlatformStats } from '@/hooks/use-prompts';
import { revalidatePath } from 'next/cache';
import type { GeneratePromptMetadataOutput } from '@/ai/flows/generate-prompt-tags';

const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const ADMIN_KEY = process.env.ADMIN_KEY;

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

export async function handleGenerateInitialPrompt(input: { user_needs: string, token: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/generate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${input.token}`,
      },
      body: JSON.stringify({ user_needs: input.user_needs }),
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

export async function handleEvaluatePrompt(input: { prompt: string; user_needs: string; token: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/evaluate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${input.token}`,
       },
      body: JSON.stringify({ prompt: input.prompt, user_needs: input.user_needs }),
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

export async function handleGetPromptSuggestions(input: { current_prompt: string; user_comments?: string; token: string }): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/suggest-improvements`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${input.token}`,
      },
      body: JSON.stringify({ current_prompt: input.current_prompt, user_comments: input.user_comments }),
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

export async function getTaskResult(status_url: string, token: string): Promise<TaskStatusResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}${status_url}`, {
        headers: { 'Authorization': `Bearer ${token}` },
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

export async function getHistoryPromptsFromDB(token: string): Promise<Prompt[]> {
  if (!token) return [];
  try {
    const response = await fetch(`${BACKEND_URL}/user/prompts`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
        },
        cache: 'no-store',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to fetch user history.');
    }

    const rawPrompts: any[] = await response.json();
    const prompts: Prompt[] = rawPrompts.map(p => ({
      id: p.id,
      userId: p.user_id, // This field is part of the UserPromptResponse model from backend
      text: p.prompt_text,
      createdAt: p.created_at,
    }));
    return prompts;
  } catch (error) {
    console.error('Failed to get history prompts:', error);
    return [];
  }
}

export async function deleteHistoryPromptFromDB(id: string, userId: string, token: string): Promise<{ success: boolean }> {
  if (!token) {
      throw new Error("Admin action requires authentication.");
  }
  if (!userId) throw new Error('User ID is required to delete a prompt.');

  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}/prompts/${id}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${token}`
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
export async function getLibraryPromptsFromDB(userId?: string): Promise<Prompt[]> {
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
    
    const prompts: Prompt[] = (await response.json()).map((p: any) => ({
      ...p,
      isStarredByUser: p.is_starred_by_user,
    }));
    return prompts;
  } catch (error) {
    console.error(`API call for library prompts failed: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function submitPromptToLibrary(request: { prompt_text: string, submission_notes?: string, token: string }): Promise<LibrarySubmission> {
    try {
      const response = await fetch(`${BACKEND_URL}/user/library/submit`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${request.token}`,
          },
          body: JSON.stringify({prompt_text: request.prompt_text, submission_notes: request.submission_notes}),
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


export async function deleteLibraryPromptFromDB(promptId: string, token: string): Promise<{ success: boolean }> {
   if (!token) {
      throw new Error("Admin action requires authentication.");
  }

  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${token}`
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

export async function toggleStarForPrompt(promptId: string, request: { user_id: string, token: string }): Promise<{ success: boolean, action: 'starred' | 'unstarred' }> {
  if (!request.user_id) throw new Error('User not authenticated.');

  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}/toggle-star`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${request.token}`,
        },
        body: JSON.stringify({user_id: request.user_id}),
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

export async function getAdminStats(token: string): Promise<PlatformStats> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/stats`, {
      headers: { 'Authorization': `Bearer ${token}` },
      cache: 'no-store',
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({ detail: `Failed to fetch admin stats. Status: ${response.status}`}));
      console.error('Failed to fetch admin stats, server responded with:', errorBody.detail);
      throw new Error(errorBody.detail);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching admin stats:', error);
    // Return a default/empty stats object on error to prevent crashing the page.
    return {
      users: { total: 0, active: 0, admins: 0 },
      prompts: { user_prompts: 0, library_prompts: 0 },
      submissions: { pending: 0 },
      tasks: { total: 0, successful: 0, success_rate: 0 },
    };
  }
}

export async function getAdminUsers(token: string): Promise<User[]> {
  if (!token) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users`, {
      headers: { 'Authorization': `Bearer ${token}` },
      cache: 'no-store',
    });
    if (!response.ok) throw new Error('Failed to fetch users.');
    const users: User[] = (await response.json()).map((u: any) => ({
      ...u,
      prompt_count: u.prompt_count || 0,
    }));
    return users;
  } catch (error) {
    console.error('Error fetching users:', error);
    return [];
  }
}

export async function getAdminLibrarySubmissions(status: 'PENDING' | 'APPROVED' | 'REJECTED', token: string): Promise<LibrarySubmission[]> {
  if (!token) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions?status=${status}`, {
      headers: { 'Authorization': `Bearer ${token}` },
      cache: 'no-store',
    });
    if (!response.ok) throw new Error('Failed to fetch library submissions.');
    
    const submissions: LibrarySubmission[] = (await response.json()).map((s: any) => ({
      id: s.id,
      prompt_text: s.prompt_text,
      user_id: s.user_id,
      status: s.status,
      submitted_at: s.created_at,
      admin_notes: s.admin_notes,
      user: {
        id: s.user_id,
        email: s.user_email || 'Unknown Email',
        full_name: s.user?.full_name || 'Unknown User',
        username: s.user?.username || '',
        is_admin: s.user?.is_admin || false,
        is_active: s.user?.is_active || true,
        created_at: s.user?.created_at || new Date().toISOString(),
        updated_at: s.user?.updated_at || new Date().toISOString(),
      },
    }));
    return submissions;
  } catch (error) {
    console.error('Error fetching library submissions:', error);
    return [];
  }
}

export async function reviewLibrarySubmission(submissionId: string, action: 'approve' | 'reject', adminNotes: string | undefined, token: string): Promise<{ success: boolean }> {
  if (!token) throw new Error("Admin action required.");
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions/${submissionId}/review`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
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
