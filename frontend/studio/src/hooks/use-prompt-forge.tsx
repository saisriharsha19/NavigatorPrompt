
'use client';

import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import { createContext, useContext, useState, ReactNode, Dispatch, SetStateAction, useEffect } from 'react';

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
  promptsGenerated: number;
  setPromptsGenerated: Dispatch<SetStateAction<number>>;
  knowledgeBase: string;
  setKnowledgeBase: Dispatch<SetStateAction<string>>;
  uploadedFiles: UploadedFile[];
  setUploadedFiles: Dispatch<SetStateAction<UploadedFile[]>>;
  fewShotExamples: string;
  setFewShotExamples: Dispatch<SetStateAction<string>>;
  scrapeUrl: string;
  setScrapeUrl: Dispatch<SetStateAction<string>>;
  sitemapUrl: string;
  setSitemapUrl: Dispatch<SetStateAction<string>>;
  includeSubdomains: boolean;
  setIncludeSubdomains: Dispatch<SetStateAction<boolean>>;
  maxSubdomains: number;
  setMaxSubdomains: Dispatch<SetStateAction<number>>;
  maxPages: number;
  setMaxPages: Dispatch<SetStateAction<number>>;
  preferSitemap: boolean;
  setPreferSitemap: Dispatch<SetStateAction<boolean>>;
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
  const [promptsGenerated, setPromptsGenerated] = useState(0);
  const [knowledgeBase, setKnowledgeBase] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [fewShotExamples, setFewShotExamples] = useState('');
  const [scrapeUrl, setScrapeUrl] = useState('');
  const [sitemapUrl, setSitemapUrl] = useState('');
  const [includeSubdomains, setIncludeSubdomains] = useState(false);
  const [maxSubdomains, setMaxSubdomains] = useState(10);
  const [maxPages, setMaxPages] = useState(100);
  const [preferSitemap, setPreferSitemap] = useState(true);
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


  // Effect to append document context to the prompt
  useEffect(() => {
    const contextMarker = '\n\n--- [DOCUMENT CONTEXT] ---';
    
    setCurrentPrompt(prevPrompt => {
      // Find and remove the old context block if it exists
      const oldContextIndex = prevPrompt.indexOf(contextMarker);
      const basePrompt = oldContextIndex !== -1 ? prevPrompt.substring(0, oldContextIndex) : prevPrompt;

      if (uploadedFiles.length === 0) {
        return basePrompt; // Return just the base prompt if no files
      }

      // Build the new context block from all uploaded files
      const newContextContent = uploadedFiles
        .map(file => `\n## Document: ${file.name}\n\n${file.content}`)
        .join('\n\n');
      
      const newContextBlock = `${contextMarker}${newContextContent}`;
      
      return basePrompt + newContextBlock;
    });
  }, [uploadedFiles, setCurrentPrompt]);


  const value = {
    userNeeds, setUserNeeds,
    currentPrompt, setCurrentPrompt,
    promptsGenerated, setPromptsGenerated,
    knowledgeBase, setKnowledgeBase,
    uploadedFiles, setUploadedFiles,
    fewShotExamples, setFewShotExamples,
    scrapeUrl, setScrapeUrl,
    sitemapUrl, setSitemapUrl,
    includeSubdomains, setIncludeSubdomains,
    maxSubdomains, setMaxSubdomains,
    maxPages, setMaxPages,
    preferSitemap, setPreferSitemap,
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
