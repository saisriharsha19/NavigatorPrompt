
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot,
  Loader2,
  Rocket,
  Sparkles,
  Clipboard,
  Check,
  Wrench,
  Upload,
  Lightbulb,
  RefreshCw,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';
import {
  handleGenerateInitialPrompt,
  handleEvaluatePrompt,
  handleGetPromptSuggestions,
  getTaskResult,
  type TaskStatusResponse,
} from '@/app/actions';
import { Badge, badgeVariants } from './ui/badge';
import { useAuth } from '@/hooks/use-auth';
import { cn } from '@/lib/utils';
import type { EvaluateAndIteratePromptOutput } from '@/ai/flows/evaluate-and-iterate-prompt';
import type { GeneratePromptSuggestionsOutput } from '@/ai/flows/get-prompt-suggestions';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { usePromptHistory } from '@/hooks/use-prompts';
import { usePromptForge } from '@/hooks/use-prompt-forge';
import { useLibrary } from '@/hooks/use-library';
import type { GenerateInitialPromptOutput } from '@/ai/flows/generate-initial-prompt';
import { Skeleton } from './ui/skeleton';
import { Separator } from './ui/separator';
import { DocumentManager } from './document-manager';

type ActionType = 'generate' | 'evaluate' | 'suggest' | null;

const formatMetricName = (name: string) => {
    const spaced = name.replace(/_score$/, '').replace(/_/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

const containerVariants = {
  hidden: { opacity: 1 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 100,
      damping: 10,
    },
  },
};

export function PromptForgeClient() {
  const { isAuthenticated, userId, login } = useAuth();
  const { addLibrarySubmission } = useLibrary();
  const { toast } = useToast();

  const {
    userNeeds, setUserNeeds,
    currentPrompt, setCurrentPrompt,
    iterationComments, setIterationComments,
    suggestions, setSuggestions,
    selectedSuggestions, setSelectedSuggestions,
    evaluationResult, setEvaluationResult,
    processingState, setProcessingState,
    taskStatusUrl, setTaskStatusUrl,
    uploadedFiles,
  } = usePromptForge();

  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const suggestionPollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text: string) => {
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text)
        .then(() => {
          setCopied(true);
          toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
          setTimeout(() => setCopied(false), 2000);
        })
        .catch(err => {
          toast({ variant: 'destructive', title: 'Copy Failed', description: 'Could not copy text.' });
          console.error('Clipboard write failed:', err);
        });
    } else {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'absolute';
      textArea.style.left = '-9999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        toast({ variant: 'destructive', title: 'Copy Failed', description: 'Could not copy text.' });
        console.error('Fallback copy failed:', err);
      } finally {
        document.body.removeChild(textArea);
      }
    }
  };
  
  const getErrorMessage = (error: unknown): string => {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  };
  
  const stopPolling = useCallback((poller: 'main' | 'suggestions') => {
    if (poller === 'main' && pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    if (poller === 'suggestions' && suggestionPollingIntervalRef.current) {
      clearInterval(suggestionPollingIntervalRef.current);
      suggestionPollingIntervalRef.current = null;
    }
  }, []);

  const pollTaskStatus = useCallback((status_url: string, actionType: ActionType) => {
    const isSuggestionTask = actionType === 'suggest';
    
    stopPolling(isSuggestionTask ? 'suggestions' : 'main');

    const intervalId = setInterval(async () => {
      try {
        const task: TaskStatusResponse = await getTaskResult(status_url);

        const statusMap: { [key: string]: string } = {
          PENDING: 'Task is pending...',
          STARTED: 'Task has started...',
          RETRY: 'Task is being retried...',
        };

        if (task.status === 'SUCCESS') {
          stopPolling(isSuggestionTask ? 'suggestions' : 'main');
          if (!isSuggestionTask) {
             setTaskStatusUrl(null);
             setProcessingState({ activeAction: null, statusText: 'Completed!' });
          } else {
             setSuggestionsLoading(false);
          }
          toast({ title: 'Success', description: `Task completed.` });
          
          if (task.result) {
            if (actionType === 'generate' && 'initial_prompt' in (task.result as any)) {
                const result = task.result as GenerateInitialPromptOutput;
                setCurrentPrompt(result.initial_prompt);
            } else if (actionType === 'evaluate' && 'improved_prompt' in (task.result as any)) {
                const result = task.result as EvaluateAndIteratePromptOutput;
                setEvaluationResult(result);
                setCurrentPrompt(result.improved_prompt);
            } else if (isSuggestionTask && Array.isArray(task.result)) {
                const result = task.result as GeneratePromptSuggestionsOutput;
                setSuggestions(result.map(s => s.description));
            }
          }

        } else if (task.status === 'FAILURE') {
          stopPolling(isSuggestionTask ? 'suggestions' : 'main');
          if (isSuggestionTask) {
            setSuggestionsLoading(false);
          } else {
            setTaskStatusUrl(null);
            setProcessingState({ activeAction: null, statusText: 'Failed!' });
          }
          toast({ variant: 'destructive', title: 'Task Failed', description: task.error_message || 'An unknown error occurred.' });
        } else {
          if (!isSuggestionTask) {
            setProcessingState(prev => ({ ...prev, statusText: statusMap[task.status] || `Processing... (${task.status.toLowerCase()})` }));
          }
        }
      } catch (error) {
        stopPolling(isSuggestionTask ? 'suggestions' : 'main');
        if (isSuggestionTask) {
          setSuggestionsLoading(false);
        } else {
          setTaskStatusUrl(null);
          setProcessingState({ activeAction: null, statusText: 'Error!' });
        }
        toast({ variant: 'destructive', title: 'Polling Error', description: getErrorMessage(error) });
      }
    }, 3000);

    if (isSuggestionTask) {
      suggestionPollingIntervalRef.current = intervalId;
    } else {
      pollingIntervalRef.current = intervalId;
    }
  }, [stopPolling, setTaskStatusUrl, setProcessingState, toast, setCurrentPrompt, setEvaluationResult, setSuggestions]);

  useEffect(() => {
    if (taskStatusUrl && processingState.activeAction) {
      pollTaskStatus(taskStatusUrl, processingState.activeAction);
    }
  }, []); 


  useEffect(() => {
    return () => {
      stopPolling('main');
      stopPolling('suggestions');
    }
  }, [stopPolling]);
  
  const onGetSuggestions = async (prompt: string, comments?: string) => {
    if (!prompt || suggestionsLoading) return;
    setSuggestionsLoading(true);
    setSuggestions([]);
    try {
      const task = await handleGetPromptSuggestions({
        current_prompt: prompt,
        user_comments: comments,
      });
      pollTaskStatus(task.status_url, 'suggest');
    } catch (error) {
      setSuggestionsLoading(false);
      toast({ variant: 'destructive', title: 'Suggestion Failed', description: getErrorMessage(error) });
    }
  };
  
  const onGenerate = async () => {
    if (!userNeeds) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please describe your assistant needs first.' });
      return;
    }
    const action: ActionType = 'generate';
    setProcessingState({ activeAction: action, statusText: 'Starting generation task...' });
    setEvaluationResult(null);
    setSuggestions([]);
    try {
      const task = await handleGenerateInitialPrompt({ user_needs: userNeeds });
      setTaskStatusUrl(task.status_url);
      setProcessingState(prev => ({ ...prev, statusText: 'Task initiated, awaiting result...' }));
      pollTaskStatus(task.status_url, action);
    } catch (error) {
      setProcessingState({ activeAction: null, statusText: '' });
      toast({ variant: 'destructive', title: 'Generation Failed', description: getErrorMessage(error) });
    }
  };

  const onEvaluate = async () => {
    if (!currentPrompt || !userNeeds) {
      toast({ variant: 'destructive', title: 'Error', description: 'A prompt and user needs are required for evaluation.' });
      return;
    }
    const action: ActionType = 'evaluate';
    setProcessingState({ activeAction: action, statusText: 'Starting evaluation task...' });
    setEvaluationResult(null);

    const docContext = uploadedFiles.length > 0 
        ? `\n\n--- [DOCUMENT CONTEXT] ---\n${uploadedFiles.map(f => `## Document: ${f.name}\n\n${f.content}`).join('\n\n')}`
        : '';
    
    const promptWithContext = `${currentPrompt}${docContext}`;

    try {
      // Send the combined prompt and the original user needs
      const task = await handleEvaluatePrompt({ prompt: promptWithContext, user_needs: userNeeds });
      setTaskStatusUrl(task.status_url);
      setProcessingState(prev => ({ ...prev, statusText: 'Task initiated, awaiting evaluation...' }));
      pollTaskStatus(task.status_url, action);
    } catch (error) {
      setProcessingState({ activeAction: null, statusText: '' });
      toast({ variant: 'destructive', title: 'Evaluation Failed', description: getErrorMessage(error) });
    }
  };
  
  const onIterate = () => {
    if (!currentPrompt || (!iterationComments && selectedSuggestions.length === 0)) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please provide feedback or select suggestions before refining.' });
      return;
    }
    const feedbackText = `
---
USER FEEDBACK FOR REFINEMENT:
Comments: ${iterationComments}
Selected Suggestions:
- ${selectedSuggestions.join('\n- ')}
---
    `;
    const newPrompt = `${currentPrompt}\n${feedbackText}`;
    setCurrentPrompt(newPrompt);

    onGetSuggestions(newPrompt, iterationComments);

    setIterationComments('');
    setSelectedSuggestions([]);
    toast({ title: 'Feedback Applied', description: 'Your feedback has been added to the prompt for re-evaluation.' });
  };
  
  const handleSuggestionToggle = (suggestion: string) => {
    setSelectedSuggestions(prev =>
      prev.includes(suggestion)
        ? prev.filter(s => s !== suggestion)
        : [...prev, suggestion]
    );
  };
  
  const onSubmitToLibrary = () => {
    if (!currentPrompt) {
        toast({
            variant: 'destructive',
            title: 'Error',
            description: 'There is no prompt to submit.',
        });
        return;
    }
    if (!userId) {
        toast({ variant: 'destructive', title: 'Authentication Error', description: 'You must be logged in to submit to the library.'});
        login();
        return;
    }
    addLibrarySubmission(currentPrompt);
  };

  const handleCreateAssistant = () => {
    if (!currentPrompt) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please generate a prompt first.' });
      return;
    }
    const docContext = uploadedFiles.length > 0 
        ? `\n\n--- [DOCUMENT CONTEXT] ---\n${uploadedFiles.map(f => `## Document: ${f.name}\n\n${f.content}`).join('\n\n')}`
        : '';
    const finalPromptForCopy = `${currentPrompt}${docContext}`;

    copyToClipboard(finalPromptForCopy);
    window.open('https://assistant.ai.it.ufl.edu/admin/assistants/new', '_blank', 'noopener,noreferrer');
    toast({
      title: 'Prompt Copied!',
      description: 'The prompt has been copied. Please paste it into the portal.',
      duration: 5000,
    });
  };
  
  return (
    <TooltipProvider>
      <motion.div
        className="flex h-full flex-col gap-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div variants={itemVariants}>
          <Card>
            <CardHeader>
              <h2 className="text-lg font-medium leading-snug">1. Describe Your Assistant</h2>
              <CardDescription>
                What are the primary goals and functionalities of your AI assistant? This will be used to generate the initial prompt.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Label htmlFor="user-needs" className="sr-only">User Needs</Label>
                <Textarea
                  id="user-needs"
                  placeholder="e.g., An assistant that helps university students find course information, check deadlines, and book appointments with advisors..."
                  value={userNeeds}
                  onChange={(e) => setUserNeeds(e.target.value)}
                  className="min-h-[120px]"
                  disabled={!!processingState.activeAction}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button onClick={onGenerate} disabled={!!processingState.activeAction || !userNeeds} variant="destructive">
                    {processingState.activeAction === 'generate' ? <Loader2 className="animate-spin" /> : <Sparkles />}
                    Generate Initial Prompt
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Generate a new prompt based on your needs.</p>
                </TooltipContent>
              </Tooltip>
            </CardFooter>
          </Card>
        </motion.div>

        <div className="grid flex-1 grid-cols-1 gap-8 lg:grid-cols-2">
          <motion.div className="flex flex-col gap-8" variants={itemVariants}>
            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium leading-snug">2. System Prompt & Context</h2>
                <CardDescription>
                  This is the generated system prompt. You can manually edit it before evaluation or refinement.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <Textarea
                    id="system-prompt"
                    placeholder="Your generated or refined prompt will appear here."
                    value={currentPrompt}
                    onChange={(e) => setCurrentPrompt(e.target.value)}
                    className="min-h-[200px] pr-12"
                    disabled={!!processingState.activeAction}
                  />
                  {currentPrompt && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute top-3 right-3 h-8 w-8"
                          onClick={() => copyToClipboard(currentPrompt)}
                          aria-label="Copy generated prompt"
                        >
                          {copied ? <Check className="text-primary" /> : <Clipboard />}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Copy prompt</p>
                      </TooltipContent>
                    </Tooltip>
                  )}
                </div>
              </CardContent>
            </Card>
            <DocumentManager />
          </motion.div>

          <motion.div className="flex flex-col gap-8" variants={itemVariants}>
            <AnimatePresence>
              {processingState.activeAction && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <Card className="shadow-lg bg-primary text-primary-foreground shadow-primary/20 dark:bg-accent dark:text-accent-foreground dark:shadow-accent/20">
                    <CardHeader className="flex-row items-center gap-4 space-y-0 p-4">
                      <div className="relative flex h-5 w-5 items-center justify-center">
                        <div className="absolute h-full w-full animate-spin rounded-full border-2 border-b-transparent border-current dark:border-current" />
                        <Loader2 className="h-3 w-3" />
                      </div>
                      <div>
                        <CardDescription className="font-medium text-inherit">
                          {processingState.statusText}
                        </CardDescription>
                      </div>
                    </CardHeader>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>

            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium leading-snug">3. Prompt Refinement</h2>
                <CardDescription>
                  Provide feedback on the current prompt to get AI-generated suggestions for improvement.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                <div className="mb-6 rounded-md border bg-muted/50 p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <p className="flex items-center text-sm font-medium">
                      <Lightbulb className="mr-2 h-5 w-5" />
                      AI Suggestions
                    </p>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => onGetSuggestions(currentPrompt)}
                          disabled={suggestionsLoading || !currentPrompt}
                          className="h-7 w-7"
                        >
                          <RefreshCw className={cn("h-4 w-4", suggestionsLoading && "animate-spin")} />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Refresh Suggestions</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  {suggestionsLoading ? (
                    <div className="space-y-2">
                      <Skeleton className="h-5 w-3/4" />
                      <Skeleton className="h-5 w-1/2" />
                      <Skeleton className="h-5 w-2/3" />
                    </div>
                  ) : suggestions.length > 0 ? (
                    <motion.div
                      className="flex flex-wrap gap-2"
                      variants={containerVariants}
                      initial="hidden"
                      animate="visible"
                    >
                      {suggestions.map((suggestion, index) => (
                        <motion.div key={index} variants={itemVariants}>
                          <button
                            type="button"
                            onClick={() => handleSuggestionToggle(suggestion)}
                            aria-pressed={selectedSuggestions.includes(suggestion)}
                            className={cn("cursor-pointer items-center transition-all hover:opacity-80 text-xs px-3 py-1 font-normal", badgeVariants({ variant: selectedSuggestions.includes(suggestion) ? 'default' : 'secondary' }))}
                          >
                            {selectedSuggestions.includes(suggestion) && (
                              <Check className="mr-1.5 h-4 w-4" />
                            )}
                            {suggestion}
                          </button>
                        </motion.div>
                      ))}
                    </motion.div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      {currentPrompt ? 'Generate or refresh suggestions.' : 'Generate a prompt to get suggestions.'}
                    </p>
                  )}
                </div>
                
                <div className="space-y-4">
                  <Label htmlFor="iteration-comments" className="text-base">Your Feedback &amp; Comments</Label>
                  <Textarea
                    id="iteration-comments"
                    placeholder="e.g., 'Make it more concise' or 'Add a rule to always ask for the user's name.'"
                    value={iterationComments}
                    onChange={(e) => setIterationComments(e.target.value)}
                    className="min-h-[100px]"
                    disabled={!!processingState.activeAction}
                  />
                </div>
              </CardContent>
              <CardFooter>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={onIterate}
                      disabled={!!processingState.activeAction || !currentPrompt || (!iterationComments && selectedSuggestions.length === 0)}
                      variant="destructive"
                    >
                      <Wrench />
                      Apply Feedback to Prompt
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Applies your feedback to the prompt text for re-evaluation.</p>
                  </TooltipContent>
                </Tooltip>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium leading-snug">4. Evaluation &amp; Deployment</h2>
                <CardDescription>
                  Review the results from our AI evaluator and deploy your agent.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                 <div className="flex items-center justify-center">
                   <Button onClick={onEvaluate} disabled={!!processingState.activeAction || !currentPrompt || !userNeeds} variant="destructive">
                       {processingState.activeAction === 'evaluate' ? <Loader2 className="animate-spin" /> : <Bot />}
                       Iterate and Evaluate
                   </Button>
                 </div>
                <div aria-live="polite" aria-atomic="true">
                  {isAuthenticated && evaluationResult ? (
                    <div className="space-y-4">
                      {evaluationResult.improvement_summary && (
                        <div className="space-y-1">
                          <h4 className="text-sm font-medium text-foreground">Improvement Summary</h4>
                          <p className="text-sm text-muted-foreground">
                            {evaluationResult.improvement_summary}
                          </p>
                        </div>
                      )}
                      
                      <Separator />

                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-foreground">Evaluation Scores</h4>
                        <ul className="space-y-1.5">
                          {(Object.keys(evaluationResult) as Array<keyof EvaluateAndIteratePromptOutput>)
                            .filter(key => key.endsWith('_score'))
                            .map((key) => {
                              const score = evaluationResult[key] as number;
                              return (
                                <li key={key} className="flex items-center justify-between">
                                  <span className="text-sm text-muted-foreground">{formatMetricName(key)}</span>
                                  {typeof score === 'number' ? (
                                    <Badge variant={score > 0.7 ? 'default' : score > 0.4 ? 'secondary' : 'destructive'} className="w-16 justify-center">
                                      {Math.round(score * 100)}%
                                    </Badge>
                                  ) : (
                                    <Badge variant="secondary" className="w-16 justify-center">N/A</Badge>
                                  )}
                                </li>
                              );
                            })}
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-8 text-center">
                      <p className="text-muted-foreground">
                        {isAuthenticated ? 'Your evaluation results will appear here.' : 'Please sign in to view evaluation results.'}
                      </p>
                      {!isAuthenticated && (
                        <Button onClick={login} className='mt-4'>Sign In</Button>
                      )}
                    </div>
                  )}
                </div>
              </CardContent>
              <CardFooter className="flex flex-col gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      className="w-full bg-accent text-accent-foreground hover:bg-accent/90 dark:bg-primary dark:text-primary-foreground dark:hover:bg-primary/90"
                      disabled={!!processingState.activeAction || !currentPrompt}
                      onClick={handleCreateAssistant}
                    >
                      <Rocket />
                      Create NaviGator Assistant
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Copies prompt and opens the NaviGator Assistant portal.</p>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      className="w-full"
                      variant="secondary"
                      disabled={!!processingState.activeAction || !currentPrompt}
                      onClick={onSubmitToLibrary}
                    >
                      <Upload />
                      Submit to Prompt Library
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Submit this prompt for admin review to be added to the public library.</p>
                  </TooltipContent>
                </Tooltip>
              </CardFooter>
            </Card>
          </motion.div>
        </div>
      </motion.div>
    </TooltipProvider>
  );
}
