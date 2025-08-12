
'use client';

import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from 'react';
import { useToast } from './use-toast';
import { deleteHistoryPromptFromDB, getHistoryPromptsFromDB } from '@/app/actions';
import { useAuth } from './use-auth';

// Represents a user in the system
export type User = {
  id: string;
  email: string;
  username: string;
  full_name: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

// Represents platform-wide statistics
export type PlatformStats = {
  total_users: number;
  total_prompts_in_history: number;
  total_prompts_in_library: number;
  pending_submissions: number;
};

export type Prompt = {
  id: string;
  userId: string;
  text: string;
  createdAt: string;
  // Library-specific fields
  stars?: number;
  summary?: string;
  tags?: string[];
  isStarredByUser?: boolean;
};

export type LibrarySubmission = {
    id: string;
    prompt_text: string;
    user_id: string;
    status: 'PENDING' | 'APPROVED' | 'REJECTED';
    submitted_at: string;
    admin_notes?: string;
    user?: User; 
};


type PromptHistoryContextType = {
  prompts: Prompt[];
  isLoading: boolean;
  deletePrompt: (id: string) => Promise<void>;
};

const PromptHistoryContext = createContext<PromptHistoryContextType | undefined>(undefined);

export function PromptHistoryProvider({ children }: { children: ReactNode }) {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const { isAuthenticated, userId, isAdmin } = useAuth();

  useEffect(() => {
    const loadPrompts = async () => {
      if (isAuthenticated && userId) {
        try {
          setIsLoading(true);
          const initialPrompts = await getHistoryPromptsFromDB(userId);
          setPrompts(initialPrompts);
        } catch (error) {
          console.error('Failed to load prompts from history', error);
          toast({
            variant: 'destructive',
            title: 'Error',
            description: 'Could not load your prompt history.',
          });
        } finally {
          setIsLoading(false);
        }
      } else {
        setPrompts([]);
        setIsLoading(false);
      }
    };
    loadPrompts();
  }, [isAuthenticated, userId, toast]);


  const deletePrompt = useCallback(async (id: string) => {
    if (!isAuthenticated || !userId) {
      toast({
        variant: 'destructive',
        title: 'Not Logged In',
        description: 'You must be logged in to delete prompts from history.',
      });
      return;
    }
    
    if (!isAdmin) {
      toast({
        variant: 'destructive',
        title: 'Permission Denied',
        description: 'Only admins can delete prompts from history.',
      });
      return;
    }

    try {
      await deleteHistoryPromptFromDB(id, userId);
      setPrompts(prev => prev.filter(p => p.id !== id));
      toast({
        title: 'Prompt Deleted',
        description: 'The prompt has been removed from the user\'s history.',
      });
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Delete Failed',
        description: error.message || 'An error occurred while deleting the prompt.',
      });
    }
  }, [isAuthenticated, userId, toast, isAdmin]);

  return (
    <PromptHistoryContext.Provider value={{ prompts, deletePrompt, isLoading }}>
      {children}
    </PromptHistoryContext.Provider>
  );
}

export function usePromptHistory() {
  const context = useContext(PromptHistoryContext);
  if (context === undefined) {
    throw new Error('usePromptHistory must be used within a PromptHistoryProvider');
  }
  return context;
}
