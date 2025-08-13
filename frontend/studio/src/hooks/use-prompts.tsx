
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
  is_student: boolean;
  is_faculty: boolean;
  is_staff: boolean;
  created_at: string;
  updated_at: string;
};

// Represents platform-wide statistics
export type PlatformStats = {
  users: {
    total: number;
    active: number;
    admins: number;
  };
  prompts: {
    user_prompts: number;
    library_prompts: number;
  };
  submissions: {
    pending: number;
  };
  tasks: {
    total: number;
    successful: number;
    success_rate: number;
  };
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
    user?: Partial<User>; 
};


type PromptHistoryContextType = {
  prompts: Prompt[];
  isLoading: boolean;
  deletePrompt: (id: string, userId: string) => Promise<void>;
};

const PromptHistoryContext = createContext<PromptHistoryContextType | undefined>(undefined);

export function PromptHistoryProvider({ children }: { children: ReactNode }) {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const { user, token } = useAuth();

  useEffect(() => {
    const loadPrompts = async () => {
      if (user && token) {
        try {
          setIsLoading(true);
          const initialPrompts = await getHistoryPromptsFromDB(token);
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
  }, [user, token, toast]);


  const deletePrompt = useCallback(async (id: string, promptUserId: string) => {
    if (!user || !token) {
      toast({
        variant: 'destructive',
        title: 'Not Logged In',
        description: 'You must be logged in to delete prompts.',
      });
      return;
    }
    
    if (!user.is_admin) {
      toast({
        variant: 'destructive',
        title: 'Permission Denied',
        description: 'Only admins can delete prompts from history.',
      });
      return;
    }

    try {
      await deleteHistoryPromptFromDB(id, promptUserId, token);
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
  }, [user, token, toast]);

  return (
    <PromptHistoryContext.Provider value={{ prompts, deletePrompt, isLoading }}>
      {children}
    </PromptHistoryContext.Provider>
  );
}

export function usePromptHistory(initialPrompts: Prompt[]) {
  const context = useContext(PromptHistoryContext);
  if (context === undefined) {
    throw new Error('usePromptHistory must be used within a PromptHistoryProvider');
  }
  
  // This allows the server-rendered prompts to be available on first load
  // while still allowing the context to manage updates.
  if (context.prompts.length === 0 && initialPrompts.length > 0 && context.isLoading) {
      context.prompts = initialPrompts;
  }
  
  return context;
}
