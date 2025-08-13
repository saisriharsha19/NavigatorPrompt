
'use client';

import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import { createContext, useContext, useState, ReactNode, Dispatch, SetStateAction } from 'react';

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
};

const PromptForgeContext = createContext<PromptForgeContextType | undefined>(undefined);

export function PromptForgeProvider({ children }: { children: ReactNode }) {
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
