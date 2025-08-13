
'use client';

import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import { createContext, useContext, useState, ReactNode, Dispatch, SetStateAction, useEffect, useCallback } from 'react';
import { useAuth } from './use-auth';

type UploadedFile = {
  id: string;
  name: string;
  content: string;
};

type ActionType = 'generate' | 'evaluate' | 'suggest' | null;

type ProcessingState = {
  activeAction: ActionType;
  statusText: string;
};

type PromptForgeContextType = {
  userNeeds: string;
  setUserNeeds: Dispatch<SetStateAction<string>>;
  currentPrompt: string;
  setCurrentPrompt: Dispatch<SetStateAction<string>>;
  uploadedFiles: UploadedFile[];
  setUploadedFiles: Dispatch<SetStateAction<UploadedFile[]>>;
  iterationComments: string;
  setIterationComments: Dispatch<SetStateAction<string>>;
  suggestions: string[];
  setSuggestions: Dispatch<SetStateAction<string[]>>;
  selectedSuggestions: string[];
  setSelectedSuggestions: Dispatch<SetStateAction<string[]>>;
  evaluationResult: EvaluateAndIteratePromptOutput | null;
  setEvaluationResult: Dispatch<SetStateAction<EvaluateAndIteratePromptOutput | null>>;
  // State for background tasks
  processingState: ProcessingState;
  setProcessingState: Dispatch<SetStateAction<ProcessingState>>;
  taskStatusUrl: string | null;
  setTaskStatusUrl: Dispatch<SetStateAction<string | null>>;
  resetForge: () => void;
};

const PromptForgeContext = createContext<PromptForgeContextType | undefined>(undefined);

export function PromptForgeProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [userNeeds, setUserNeeds] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [iterationComments, setIterationComments] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedSuggestions, setSelectedSuggestions] = useState<string[]>([]);
  const [evaluationResult, setEvaluationResult] = useState<EvaluateAndIteratePromptOutput | null>(null);

  // New state for managing background tasks globally
  const [processingState, setProcessingState] = useState<ProcessingState>({
    activeAction: null,
    statusText: '',
  });
  const [taskStatusUrl, setTaskStatusUrl] = useState<string | null>(null);

  const resetForge = useCallback(() => {
    setUserNeeds('');
    setCurrentPrompt('');
    setUploadedFiles([]);
    setIterationComments('');
    setSuggestions([]);
    setSelectedSuggestions([]);
    setEvaluationResult(null);
    setProcessingState({ activeAction: null, statusText: '' });
    setTaskStatusUrl(null);
  }, []);

  useEffect(() => {
    // If the user logs out, reset the state of the forge.
    if (!user) {
      resetForge();
    }
  }, [user, resetForge]);


  const value = {
    userNeeds, setUserNeeds,
    currentPrompt, setCurrentPrompt,
    uploadedFiles, setUploadedFiles,
    iterationComments, setIterationComments,
    suggestions, setSuggestions,
    selectedSuggestions, setSelectedSuggestions,
    evaluationResult, setEvaluationResult,
    // Provide task state to context
    processingState, setProcessingState,
    taskStatusUrl, setTaskStatusUrl,
    resetForge,
  };
  
  return (
    <PromptForgeContext.Provider value={value}>
      {children}
    </PromptForgeContext.Provider>
  );
}

export function usePromptForge() {
  const context = useContext(PromptForgeContext);
  if (context === undefined) {
    throw new Error('usePromptForge must be used within a PromptForgeProvider');
  }
  return context;
}
