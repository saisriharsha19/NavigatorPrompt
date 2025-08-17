
'use client';

import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from 'react';
import { useToast } from './use-toast';
import { useAuth } from './use-auth';
import type { Prompt } from './use-prompts';
import Cookies from 'js-cookie';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';

type LibraryContextType = {
  libraryPrompts: Prompt[];
  isLoading: boolean;
  addLibraryPrompt: (text: string, notes?: string) => Promise<void>;
  toggleStar: (id: string) => Promise<void>;
  deleteLibraryPrompt: (id: string) => Promise<void>;
};

const LibraryContext = createContext<LibraryContextType | undefined>(undefined);

const POLLING_INTERVAL = 15000; // Poll every 15 seconds

export function LibraryProvider({ children }: { children: ReactNode }) {
  const [libraryPrompts, setLibraryPrompts] = useState<Prompt[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const { isAuthenticated, userId, isAdmin } = useAuth();

  const getAuthHeaders = useCallback(() => {
    const token = Cookies.get('auth_token');
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  }, []);

  const loadPrompts = useCallback(async (isInitialLoad = false) => {
    if (isInitialLoad) setIsLoading(true);

    try {
      const url = new URL(`${BACKEND_URL}/library/prompts`);
      if (userId) {
        url.searchParams.append('user_id', userId);
      }
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: getAuthHeaders(),
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`Failed to load library prompts: ${response.statusText}`);
      }
      
      const prompts: Prompt[] = await response.json();
      setLibraryPrompts(prompts);
    } catch (error) {
      console.error('Failed to load prompts from library', error);
      if (isInitialLoad) {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: 'Could not load the prompt library. Please check your connection.',
        });
      }
    } finally {
      if (isInitialLoad) setIsLoading(false);
    }
  }, [toast, userId, getAuthHeaders]);

  useEffect(() => {
    loadPrompts(true); // Initial load
    
    const interval = setInterval(() => {
      loadPrompts(false); // Subsequent polls
    }, POLLING_INTERVAL);

    // Clean up the interval on component unmount
    return () => clearInterval(interval);
  }, [loadPrompts]);

  const addLibraryPrompt = useCallback(async (text: string, notes?: string) => {
    if (!isAuthenticated || !userId) {
      toast({
        variant: 'destructive',
        title: 'Authentication Required',
        description: 'You must be signed in to submit prompts to the library.',
      });
      return;
    }
    
    if (!text.trim()) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Prompt text cannot be empty.',
      });
      return;
    }
    
    try {
      const response = await fetch(`${BACKEND_URL}/user/library/submit`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          prompt_text: text,
          submission_notes: notes || null,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to submit prompt.' }));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }
      
      toast({
        title: 'Prompt Submitted',
        description: 'Your prompt has been submitted for admin review and will appear in the library once approved.',
      });
      
      // Re-fetch the entire list to ensure correct order and data.
      await loadPrompts(false);
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Submission Failed',
        description: error.message || 'An error occurred while submitting the prompt.',
      });
      throw error;
    }
  }, [isAuthenticated, userId, toast, loadPrompts, getAuthHeaders]);

  const deleteLibraryPrompt = useCallback(async (id: string) => {
    if (!isAuthenticated || !isAdmin) {
      toast({
        variant: 'destructive',
        title: 'Permission Denied',
        description: 'You do not have permission to delete library prompts.',
      });
      return;
    }

    const originalPrompts = [...libraryPrompts];
    setLibraryPrompts(prev => prev.filter(p => p.id !== id));

    try {
      const response = await fetch(`${BACKEND_URL}/library/prompts/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });

      if (response.status === 204) {
        toast({
          title: 'Prompt Deleted',
          description: 'The prompt has been removed from the library.',
        });
        return;
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to delete prompt.' }));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }

    } catch (error: any) {
      setLibraryPrompts(originalPrompts); // Revert on error
      toast({
        variant: 'destructive',
        title: 'Delete Failed',
        description: error.message || 'An error occurred while deleting the prompt.',
      });
    }
  }, [isAuthenticated, isAdmin, libraryPrompts, toast, getAuthHeaders]);

  const toggleStar = useCallback(async (promptId: string) => {
    if (!isAuthenticated || !userId) {
        toast({ 
          variant: 'destructive', 
          title: 'Authentication Required',
          description: 'Please sign in to star prompts.' 
        });
        return;
    }

    const originalPrompts = [...libraryPrompts];
    let optimisticAction: 'starred' | 'unstarred' | undefined;

    // Optimistic update
    setLibraryPrompts(prev =>
      prev.map(p => {
        if (p.id === promptId) {
          const isStarred = !!p.isStarredByUser;
          optimisticAction = isStarred ? 'unstarred' : 'starred';
          const currentStars = p.stars ?? 0;
          return {
            ...p,
            isStarredByUser: !isStarred,
            stars: isStarred ? Math.max(0, currentStars - 1) : currentStars + 1,
          };
        }
        return p;
      })
    );

    try {
        const response = await fetch(`${BACKEND_URL}/library/prompts/${promptId}/toggle-star`, {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({ user_id: userId }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to toggle star.' }));
          throw new Error(errorData.detail || `Request failed with status ${response.status}`);
        }
        
        const result = await response.json();
        
        // If the optimistic action was wrong, resync state
        if (result.action !== optimisticAction) {
           await loadPrompts(false);
        }
    } catch (error: any) {
        toast({
            variant: 'destructive',
            title: 'Error',
            description: error.message || 'Failed to update star status.',
        });
        // Revert optimistic update on failure
        setLibraryPrompts(originalPrompts);
    }
  }, [isAuthenticated, userId, toast, libraryPrompts, loadPrompts, getAuthHeaders]);

  return (
    <LibraryContext.Provider value={{ 
      libraryPrompts, 
      addLibraryPrompt, 
      toggleStar, 
      isLoading, 
      deleteLibraryPrompt 
    }}>
      {children}
    </LibraryContext.Provider>
  );
}

export function useLibrary() {
  const context = useContext(LibraryContext);
  if (context === undefined) {
    throw new Error('useLibrary must be used within a LibraryProvider');
  }
  return context;
}
