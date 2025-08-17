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

// --- Helper to get auth headers from request ---
function getAuthHeaders(token?: string): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  return headers;
}


// Add this helper function at the top of your actions.ts file, after getAuthHeaders

function getAdminAuthHeaders(token?: string): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  // Add admin key for development - replace with your actual admin key from backend settings
  if (process.env.NODE_ENV === 'development') {
    // Check your backend config.py for the actual ADMIN_KEY value
    headers['X-Admin-Key'] = process.env.NEXT_PUBLIC_ADMIN_KEY || 'dev-admin-key-123';
  }
  
  return headers;
}

// Then update these specific functions to use getAdminAuthHeaders instead of getAuthHeaders:

export async function getAdminStats(token?: string): Promise<{
  users: { total: number; active: number; admins: number };
  prompts: { user_prompts: number; library_prompts: number };
  submissions: { pending: number };
  tasks: { total: number; successful: number; success_rate: number };
}> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/stats`, {
      method: 'GET',
      headers: getAdminAuthHeaders(token), // Changed this line
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch admin stats.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch admin stats: ${getErrorMessage(error)}`);
    return {
      users: { total: 0, active: 0, admins: 0 },
      prompts: { user_prompts: 0, library_prompts: 0 },
      submissions: { pending: 0 },
      tasks: { total: 0, successful: 0, success_rate: 0 }
    };
  }
}

export async function getAdminUsers(
  token?: string,
  params?: {
    skip?: number;
    limit?: number;
    role_filter?: string;
    search?: string;
  }
): Promise<AdminUser[]> {
  try {
    const url = new URL(`${BACKEND_URL}/admin/users`);
    
    if (params?.skip !== undefined) url.searchParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) url.searchParams.append('limit', params.limit.toString());
    if (params?.role_filter) url.searchParams.append('role_filter', params.role_filter);
    if (params?.search) url.searchParams.append('search', params.search);
    
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: getAdminAuthHeaders(token), // Changed this line
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch admin users.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch admin users: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function getAdminLibrarySubmissions(
  token?: string,
  params?: {
    status_filter?: string;
    skip?: number;
    limit?: number;
  }
): Promise<LibrarySubmission[]> {
  try {
    const url = new URL(`${BACKEND_URL}/admin/library/submissions`);
    
    if (params?.status_filter) url.searchParams.append('status_filter', params.status_filter);
    if (params?.skip !== undefined) url.searchParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) url.searchParams.append('limit', params.limit.toString());
    
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: getAdminAuthHeaders(token), // Changed this line
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch library submissions.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch library submissions: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function reviewLibrarySubmission(
  submissionId: string,
  action: 'approve' | 'reject',
  adminNotes?: string,
  token?: string
): Promise<{ success: boolean; action: string; submission_id: string }> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions/${submissionId}/review`, {
      method: 'POST',
      headers: getAdminAuthHeaders(token), // Changed this line
      body: JSON.stringify({
        action,
        admin_notes: adminNotes || null,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to review submission.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to review submission: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}




// --- Task-based API Calls ---

export type TaskCreationResponse = {
  task_id: string;
  status_url: string;
};

export async function handleGenerateInitialPrompt(
  input: { user_needs: string }, 
  token?: string
): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/generate`, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify(input),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to start generation task.' }));
      throw new Error(errorData.detail || 'Failed to start generation task.');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in handleGenerateInitialPrompt:', error);
    throw new Error(`An error occurred while starting the generation task: ${getErrorMessage(error)}`);
  }
}

export async function handleEvaluatePrompt(
  input: { prompt: string; user_needs: string; }, 
  token?: string
): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/evaluate`, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify(input),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to start evaluation task.' }));
      throw new Error(errorData.detail || 'Failed to start evaluation task.');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in handleEvaluatePrompt:', error);
    throw new Error(`An error occurred while starting the evaluation task: ${getErrorMessage(error)}`);
  }
}



export async function handleGetPromptSuggestions(
  input: { current_prompt: string; user_comments?: string }, 
  token?: string
): Promise<TaskCreationResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/prompts/suggest-improvements`, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify(input),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to start suggestion task.' }));
      throw new Error(errorData.detail || 'Failed to start suggestion task.');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in handleGetPromptSuggestions:', error);
    throw new Error(`An error occurred while starting the suggestion task: ${getErrorMessage(error)}`);
  }
}


// --- Admin User Management Actions ---

export type AdminUser = {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  first_name?: string;
  last_name?: string;
  student_id?: string;
  affiliation?: string;
  is_active: boolean;
  is_admin: boolean;
  is_student: boolean;
  is_faculty: boolean;
  is_staff: boolean;
  last_login?: string;
  created_at: string;
  prompt_count: number;
};


export async function createAdminUser(
  userData: {
    email: string;
    username: string;
    full_name?: string;
    first_name?: string;
    last_name?: string;
    student_id?: string;
    affiliation?: string;
    is_admin?: boolean;
    is_student?: boolean;
    is_faculty?: boolean;
    is_staff?: boolean;
  },
  token?: string
): Promise<AdminUser> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users`, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to create user.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to create user: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}

export async function updateAdminUser(
  userId: string,
  userData: Partial<{
    email: string;
    username: string;
    full_name: string;
    first_name: string;
    last_name: string;
    student_id: string;
    affiliation: string;
    is_active: boolean;
    is_admin: boolean;
    is_student: boolean;
    is_faculty: boolean;
    is_staff: boolean;
  }>,
  token?: string
): Promise<AdminUser> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}`, {
      method: 'PUT',
      headers: getAuthHeaders(token),
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to update user.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to update user: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}

export async function deleteAdminUser(userId: string, token?: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(token),
    });

    if (response.status === 204) {
      return { success: true };
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to delete user.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    return { success: true };
  } catch (error) {
    console.error(`Failed to delete user: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}

// --- Admin Library Submission Management Actions ---

export type LibrarySubmission = {
  id: string;
  user_id: string;
  prompt_text: string;
  submission_notes?: string;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
  admin_notes?: string;
  reviewed_by?: string;
  reviewed_at?: string;
  created_at: string;
  summary?: string;
  tags?: string[];
  user_email?: string;
};


export async function deleteLibrarySubmission(submissionId: string, token?: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/library/submissions/${submissionId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(token),
    });

    if (response.status === 204) {
      return { success: true };
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to delete submission.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    return { success: true };
  } catch (error) {
    console.error(`Failed to delete submission: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}

// --- User Prompt Management Actions ---

export type UserPromptHistory = {
  id: string;
  prompt_text: string;
  task_type: string;
  created_at: string;
  summary?: string;
  tags?: string[];
};

export async function getUserPrompts(userId: string, token?: string): Promise<UserPromptHistory[]> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}/prompts`, {
      method: 'GET',
      headers: getAuthHeaders(token),
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch user prompts.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch user prompts: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function deleteUserPrompt(userId: string, promptId: string, token?: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${BACKEND_URL}/admin/users/${userId}/prompts/${promptId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(token),
    });

    if (response.status === 204) {
      return { success: true };
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to delete user prompt.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    return { success: true };
  } catch (error) {
    console.error(`Failed to delete user prompt: ${getErrorMessage(error)}`);
    throw new Error(getErrorMessage(error));
  }
}


// This function is kept for conceptual mapping, but the backend doesn't have a direct "iterate" endpoint.
// Iteration is now: get suggestions -> user selects -> client-side prompt update -> re-evaluate.
export async function handleIterateOnPrompt(input: { currentPrompt: string; userComments: string; }): Promise<{ newPrompt: string }> {
  console.warn("handleIterateOnPrompt is a client-side operation now and should be refactored.");
  // This would be handled client-side by applying suggestions.
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

export async function getTaskResult(status_url: string, token?: string): Promise<TaskStatusResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}${status_url}`, {
      headers: getAuthHeaders(token),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Failed to get task status from ${status_url}.` }));
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

// Replace the Database Actions section in your actions.ts with this:

// --- User Prompt History Actions (Backend API) ---

// Replace the existing history functions with these:

export async function getHistoryPromptsFromDB(userId: string): Promise<Prompt[]> {
  if (!userId) return [];
  try {
    const url = new URL(`${BACKEND_URL}/history/prompts`);
    url.searchParams.append('user_id', userId);
    
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
    console.error('Failed to get history prompts:', error);
    return [];
  }
}

export async function addHistoryPromptToDB(promptText: string, userId: string): Promise<Prompt> {
  if (!userId) throw new Error('User not authenticated.');

  try {
    const response = await fetch(`${BACKEND_URL}/history/prompts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt_text: promptText, user_id: userId }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    const newPrompt: Prompt = await response.json();
    revalidatePath('/history');
    return newPrompt;
  } catch (error: any) {
    console.error('Failed to add prompt to history via API:', error);
    throw new Error(error.message || 'Failed to save prompt to history.');
  }
}

export async function deleteHistoryPromptFromDB(id: string, userId: string): Promise<{ success: boolean }> {
  if (!userId) throw new Error('User not authenticated.');

  try {
    const url = new URL(`${BACKEND_URL}/history/prompts/${id}`);
    url.searchParams.append('user_id', userId);
    
    const response = await fetch(url.toString(), {
      method: 'DELETE',
    });

    if (response.status === 204) {
      revalidatePath('/history');
      return { success: true };
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to parse API error response.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    revalidatePath('/history');
    return { success: true };
  } catch (error: any) {
    console.error('Failed to delete prompt from history via API:', error);
    throw new Error(error.message || 'Failed to delete prompt from history.');
  }
}

// --- Library Actions (API only) ---
export async function getLibraryPromptsFromDB(userId: string | null, token?: string): Promise<Prompt[]> {
  try {
    const url = new URL(`${BACKEND_URL}/library/prompts`);
    if (userId) {
      url.searchParams.append('user_id', userId);
    }
    
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: getAuthHeaders(token),
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch library prompts.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    const prompts: Prompt[] = await response.json();
    return prompts;
  } catch (error) {
    console.error(`API call for library prompts failed: ${getErrorMessage(error)}`);
    return [];
  }
}

export async function addLibraryPromptToDB(
  request: { prompt_text: string, user_id: string }, 
  token?: string
): Promise<Prompt> {
  if (!request.user_id) throw new Error('User not authenticated.');

  try {
    const response = await fetch(`${BACKEND_URL}/user/library/submit`, {
        method: 'POST',
        headers: getAuthHeaders(token),
        body: JSON.stringify({
          prompt_text: request.prompt_text,
          submission_notes: null,
        }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to submit prompt to library.' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }
    
    const result = await response.json();
    revalidatePath('/library');
    
    // Return a mock prompt object since the submission goes through review
    return {
      id: crypto.randomUUID(),
      userId: request.user_id,
      text: request.prompt_text,
      createdAt: new Date().toISOString(),
    };

  } catch (error: any) {
    console.error('Failed to add prompt to library via API:', error);
    throw new Error(error.message || 'Failed to submit prompt to library.');
  }
}

export async function deleteLibraryPromptFromDB(promptId: string, token?: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}`, {
        method: 'DELETE',
        headers: getAuthHeaders(token),
    });

    if (response.status === 204) {
      revalidatePath('/library');
      return { success: true };
    }
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to delete prompt from library.' }));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    revalidatePath('/library');
    return { success: true };
  } catch (error: any) {
    console.error('Failed to delete prompt from library via API:', error);
    throw new Error(error.message || 'Failed to delete prompt from library.');
  }
}

export async function toggleStarForPrompt(
  promptId: string, 
  request: { user_id: string }, 
  token?: string
): Promise<{ success: boolean, action: 'starred' | 'unstarred' }> {
  if (!request.user_id) throw new Error('User not authenticated.');

  try {
    const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}/toggle-star`, {
        method: 'POST',
        headers: getAuthHeaders(token),
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to toggle star for prompt.' }));
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