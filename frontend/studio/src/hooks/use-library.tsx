
'use client';

import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from 'react';
import { useToast } from './use-toast';
import { submitPromptToLibrary, getLibraryPromptsFromDB, toggleStarForPrompt, deleteLibraryPromptFromDB } from '@/app/actions';
import { useAuth } from './use-auth';
import type { Prompt } from './use-prompts';


type LibraryContextType = {
  libraryPrompts: Prompt[];
  isLoading: boolean;
  addLibrarySubmission: (text: string) => Promise<void>;
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

  const loadPrompts = useCallback(async (isInitialLoad = false) => {
    if (isInitialLoad) setIsLoading(true);

    try {
      const initialPrompts = await getLibraryPromptsFromDB(userId);
      setLibraryPrompts(initialPrompts);
    } catch (error) {
      console.error('Failed to load prompts from library', error);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Could not load the prompt library. The API might be down.',
      });
    } finally {
      if (isInitialLoad) setIsLoading(false);
    }
  }, [toast, userId]);


  useEffect(() => {
    loadPrompts(true); // Initial load
    
    const interval = setInterval(() => {
      loadPrompts(false); // Subsequent polls
    }, POLLING_INTERVAL);

    return () => clearInterval(interval);
  }, [loadPrompts]);

  const addLibrarySubmission = useCallback(async (text: string) => {
    if (!isAuthenticated || !userId) {
      toast({
        variant: 'destructive',
        title: 'Authentication Required',
        description: 'You must be signed in to submit prompts to the public library.',
      });
      return;
    }
    
    if (!text.trim()) return;
    
    try {
      await submitPromptToLibrary({ prompt_text: text });
      toast({
        title: 'Prompt Submitted to Library',
        description: 'Your prompt has been sent for admin approval.',
      });
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Submission Failed',
        description: error.message || 'An error occurred while submitting the prompt.',
      });
    }
  }, [isAuthenticated, userId, toast]);

  const deleteLibraryPrompt = useCallback(async (id: string) => {
    if (!isAuthenticated || !userId || !isAdmin) {
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
      await deleteLibraryPromptFromDB(id);
      toast({
        title: 'Prompt Deleted',
        description: 'The prompt has been removed from the library.',
      });
    } catch (error: any) {
      setLibraryPrompts(originalPrompts); // Revert on error
      toast({
        variant: 'destructive',
        title: 'Delete Failed',
        description: error.message || 'An error occurred while deleting the prompt.',
      });
    }
  }, [isAuthenticated, userId, isAdmin, libraryPrompts, toast]);


  const toggleStar = useCallback(async (promptId: string) => {
    if (!isAuthenticated || !userId) {
        toast({ variant: 'destructive', title: 'Please sign in to star prompts.' });
        return;
    }

    const originalPrompts = [...libraryPrompts];
    let optimisticAction: 'starred' | 'unstarred' | undefined;

    setLibraryPrompts(prev =>
      prev.map(p => {
        if (p.id === promptId) {
          const isStarred = !!p.isStarredByUser;
          optimisticAction = isStarred ? 'unstarred' : 'starred';
          const currentStars = p.stars ?? 0;
          return {
            ...p,
            isStarredByUser: !isStarred,
            stars: isStarred ? currentStars - 1 : currentStars + 1,
          };
        }
        return p;
      })
    );

    try {
        const result = await toggleStarForPrompt(promptId, { user_id: userId });
        if (result.action !== optimisticAction) {
           await loadPrompts(false); 
        }
    } catch (error: any) {
        toast({
            variant: 'destructive',
            title: 'Error',
            description: error.message,
        });
        setLibraryPrompts(originalPrompts);
    }
  }, [isAuthenticated, userId, toast, libraryPrompts, loadPrompts]);

  return (
    <LibraryContext.Provider value={{ libraryPrompts, addLibrarySubmission, toggleStar, isLoading, deleteLibraryPrompt }}>
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

    