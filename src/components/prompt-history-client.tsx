
'use client';

import { useState } from 'react';
import { usePromptHistory, type Prompt } from '@/hooks/use-prompts';
import { Button, buttonVariants } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Clipboard, Check, Trash2, Search, UserCircle, Lock, Eye } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { formatDistanceToNow } from 'date-fns';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { useAuth } from '@/hooks/use-auth';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { cn } from '@/lib/utils';

function PromptCardSkeleton() {
  return (
    <Card className="flex flex-col">
      <CardHeader>
        <Skeleton className="h-5 w-2/3" />
        <Skeleton className="h-4 w-1/3 mt-2" />
      </CardHeader>
      <CardContent className="flex-grow space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-4/5" />
      </CardContent>
      <CardFooter className="flex justify-end gap-2">
        <Skeleton className="h-8 w-8 rounded-full" />
        <Skeleton className="h-8 w-8 rounded-full" />
        <Skeleton className="h-8 w-8 rounded-full" />
      </CardFooter>
    </Card>
  );
}

export function PromptHistoryClient() {
  const { prompts, deletePrompt, isLoading } = usePromptHistory();
  const { isAuthenticated, login } = useAuth();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewingPrompt, setViewingPrompt] = useState<Prompt | null>(null);
  const { toast } = useToast();

  const copyToClipboard = (prompt: Prompt) => {
    navigator.clipboard.writeText(prompt.text);
    setCopiedId(prompt.id);
    toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
    setTimeout(() => setCopiedId(null), 2000);
  };
  
  const filteredPrompts = prompts.filter(prompt => 
    prompt.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getPromptTitle = (text: string) => {
    const words = text.split(' ');
    const title = words.slice(0, 8).join(' ');
    return words.length > 8 ? `${title}...` : title;
  };
  
  if (!isAuthenticated) {
    return (
      <div className="container mx-auto max-w-7xl py-8 px-4 sm:px-6 lg:px-8">
        <div className="flex h-[60vh] flex-col items-center justify-center rounded-lg border-2 border-dashed bg-muted/50 p-12 text-center">
            <Lock className="h-16 w-16 text-muted-foreground" />
            <h2 className="mt-6 text-2xl font-semibold tracking-tight">Access Your Work History</h2>
            <p className="mt-2 text-muted-foreground">
              Sign in to view, save, and manage your personal prompt history.
            </p>
            <Button onClick={login} className="mt-6">
                <UserCircle className="mr-2 h-5 w-5" />
                Sign In
            </Button>
        </div>
      </div>
    )
  }


  return (
    <TooltipProvider>
      <Dialog open={!!viewingPrompt} onOpenChange={(isOpen) => !isOpen && setViewingPrompt(null)}>
        <DialogContent className="sm:max-w-2xl">
            <DialogHeader>
                <DialogTitle>{viewingPrompt ? getPromptTitle(viewingPrompt.text) : 'Prompt Details'}</DialogTitle>
                <DialogDescription>
                    Full content of the selected prompt. You can copy it from here.
                </DialogDescription>
            </DialogHeader>
            <div className="relative p-6">
                <ScrollArea className="h-96 w-full rounded-md border bg-muted/50 p-4">
                    <p className="text-sm text-foreground whitespace-pre-wrap">
                        {viewingPrompt?.text}
                    </p>
                </ScrollArea>
                {viewingPrompt && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="absolute top-8 right-12 h-10 w-10 hover:bg-accent hover:text-accent-foreground"
                                onClick={() => copyToClipboard(viewingPrompt)}
                                aria-label="Copy prompt"
                            >
                                {copiedId === viewingPrompt.id ? <Check className="h-6 w-6 text-primary" /> : <Clipboard className="h-6 w-6" />}
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            <p>Copy prompt</p>
                        </TooltipContent>
                    </Tooltip>
                )}
            </div>
        </DialogContent>
      </Dialog>
      <div className="container mx-auto max-w-7xl py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-8 flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="rounded-lg bg-muted p-6 w-full sm:w-auto">
            <div className='flex items-center gap-3 sm:gap-4'>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-6 w-6 shrink-0 text-accent dark:text-primary sm:h-8 sm:w-8"
                aria-hidden="true"
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M12 7v5l3 3" />
              </svg>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-primary dark:text-accent sm:text-3xl">Prompt History</h1>
                <p className="text-sm text-muted-foreground">
                  Your 20 most recent prompts are saved automatically.
                </p>
              </div>
            </div>
          </div>
          <div className="relative w-full sm:w-80">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search history..."
              aria-label="Search prompt history"
              className="h-11 w-full rounded-full border-transparent bg-muted pl-12 pr-4 transition-colors focus:bg-background focus:ring-2 focus:ring-ring"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <ScrollArea className="h-full">
          {isLoading ? (
            <ul className="grid grid-cols-1 gap-6">
              {Array.from({ length: 8 }).map((_, i) => <li key={i}><PromptCardSkeleton /></li>)}
            </ul>
          ) : filteredPrompts.length > 0 ? (
            <ul className="grid grid-cols-1 gap-6">
              {filteredPrompts.map((prompt) => (
                <li key={prompt.id}>
                  <Card className="flex flex-col">
                    <CardHeader className="p-4 sm:p-6">
                      <CardTitle>
                        {getPromptTitle(prompt.text)}
                      </CardTitle>
                      <CardDescription>
                        Saved{' '}
                        {formatDistanceToNow(new Date(prompt.createdAt), {
                          addSuffix: true,
                        })}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow p-4 pt-0 sm:p-6">
                      <p className="line-clamp-6 text-sm text-foreground/80">
                        {prompt.text}
                      </p>
                    </CardContent>
                    <CardFooter className="flex justify-end gap-2 p-4 pt-0 sm:p-6">
                       <Tooltip>
                        <TooltipTrigger asChild>
                           <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => setViewingPrompt(prompt)}
                                aria-label="View prompt"
                            >
                                <Eye className="h-4 w-4" />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>View full prompt</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => copyToClipboard(prompt)}
                            aria-label="Copy prompt"
                          >
                            {copiedId === prompt.id ? (
                              <Check className="h-4 w-4 text-primary" />
                            ) : (
                              <Clipboard className="h-4 w-4" />
                            )}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Copy prompt</p>
                        </TooltipContent>
                      </Tooltip>

                      {isAuthenticated && (
                        <AlertDialog>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <AlertDialogTrigger asChild>
                                  <Button variant="ghost" size="icon" className="text-destructive/70 hover:bg-accent hover:text-accent-foreground focus-visible:bg-accent focus-visible:text-accent-foreground focus-visible:ring-accent" aria-label="Delete prompt">
                                      <Trash2 className="h-4 w-4 transition-colors" />
                                  </Button>
                              </AlertDialogTrigger>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Delete prompt</p>
                            </TooltipContent>
                          </Tooltip>
                          <AlertDialogContent>
                            <AlertDialogHeader className="text-center sm:text-left">
                              <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                              <AlertDialogDescription>
                                This action cannot be undone. This will permanently delete this prompt from your history.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel className={cn(buttonVariants({variant: 'default'}))}>Cancel</AlertDialogCancel>
                              <AlertDialogAction onClick={() => deletePrompt(prompt.id)} className={cn(buttonVariants({variant: 'destructive'}))}>
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      )}
                    </CardFooter>
                  </Card>
                </li>
              ))}
            </ul>
          ) : (
            <div role="region" aria-labelledby="empty-history-heading" className="flex h-[50vh] flex-col items-center justify-center rounded-lg border-2 border-dashed">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mb-4 h-16 w-16 text-muted-foreground"
                aria-hidden="true"
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M12 7v5l3 3" />
              </svg>
              <h2 id="empty-history-heading" className="text-2xl font-semibold">History is Empty</h2>
              <p className="mt-2 text-muted-foreground">
                {searchQuery ? "No prompts match your search." : "Go to the generator to create your first prompt!"}
              </p>
            </div>
          )}
        </ScrollArea>
      </div>
    </TooltipProvider>
  );
}
