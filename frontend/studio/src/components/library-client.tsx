'use client';

import { useState } from 'react';
import type { Prompt } from '@/hooks/use-prompts';
import { useLibrary } from '@/hooks/use-library';
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
import { Clipboard, Check, Search, Star, Trash2, Eye, CheckCircle, XCircle, Clock, Upload } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { useAuth } from '@/hooks/use-auth';
import { cn } from '@/lib/utils';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { formatDistanceToNow } from 'date-fns';

// Extended Prompt type to include submission data
interface ExtendedPrompt extends Prompt {
  status?: 'APPROVED' | 'PENDING' | 'REJECTED';
  admin_notes?: string;
  submission_notes?: string;
}

function PromptCardSkeleton() {
  return (
    <Card className="flex flex-col">
      <CardHeader>
        <Skeleton className="h-5 w-3/4" />
        <Skeleton className="h-4 w-1/4 mt-2" />
      </CardHeader>
      <CardContent className="flex-grow space-y-4">
        <div className="flex flex-wrap gap-1.5">
            <Skeleton className="h-5 w-14 rounded-full" />
            <Skeleton className="h-5 w-20 rounded-full" />
            <Skeleton className="h-5 w-16 rounded-full" />
        </div>
        <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
        </div>
      </CardContent>
      <CardFooter className="flex items-center justify-between">
        <Skeleton className="h-8 w-16" />
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
      </CardFooter>
    </Card>
  );
}

function getStatusBadge(status?: string) {
  switch (status) {
    case 'APPROVED':
      return (
        <Badge variant="default" className="flex items-center gap-1">
          <CheckCircle className="h-3 w-3" />
          Approved
        </Badge>
      );
    case 'PENDING':
      return (
        <Badge variant="secondary" className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          Pending Review
        </Badge>
      );
    case 'REJECTED':
      return (
        <Badge variant="destructive" className="flex items-center gap-1">
          <XCircle className="h-3 w-3" />
          Rejected
        </Badge>
      );
    default:
      return null;
  }
}

export function LibraryClient() {
  const { libraryPrompts, toggleStar, isLoading, deleteLibraryPrompt, addLibraryPrompt } = useLibrary();
  const { isAuthenticated, isAdmin } = useAuth();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewingPrompt, setViewingPrompt] = useState<ExtendedPrompt | null>(null);
  const [submissionDialogOpen, setSubmissionDialogOpen] = useState(false);
  const [submissionText, setSubmissionText] = useState('');
  const [submissionNotes, setSubmissionNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = (prompt: ExtendedPrompt) => {
    navigator.clipboard.writeText(prompt.text);
    setCopiedId(prompt.id);
    toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
    setTimeout(() => setCopiedId(null), 2000);
  };
  
  const filteredPrompts = libraryPrompts.filter(prompt => {
    const query = searchQuery.toLowerCase();
    const textMatch = prompt.text.toLowerCase().includes(query);
    const summaryMatch = prompt.summary?.toLowerCase().includes(query);
    const tagsMatch = prompt.tags?.some(tag => tag.toLowerCase().includes(query));
    return textMatch || summaryMatch || tagsMatch;
  });
  
  const isValidDate = (date: any) => date && !isNaN(new Date(date).getTime());

  const handleSubmitToLibrary = async () => {
    if (!submissionText.trim()) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Please enter a prompt to submit.',
      });
      return;
    }

    setIsSubmitting(true);
    try {
      await addLibraryPrompt(submissionText, submissionNotes);
      setSubmissionText('');
      setSubmissionNotes('');
      setSubmissionDialogOpen(false);
      toast({
        title: 'Submission Successful',
        description: 'Your prompt has been submitted for admin review.',
      });
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Submission Failed',
        description: error.message || 'Failed to submit prompt for review.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <TooltipProvider>
      {/* View Prompt Dialog */}
      <Dialog open={!!viewingPrompt} onOpenChange={(isOpen) => !isOpen && setViewingPrompt(null)}>
          <DialogContent className="sm:max-w-2xl">
              <DialogHeader>
                  <div className="flex items-center justify-between">
                    <DialogTitle>{viewingPrompt?.summary || 'Prompt Details'}</DialogTitle>
                    {viewingPrompt?.status && getStatusBadge(viewingPrompt.status)}
                  </div>
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
                  {viewingPrompt?.admin_notes && (
                    <div className="mt-4 p-3 bg-muted rounded-md">
                      <h4 className="text-sm font-medium mb-1">Admin Notes:</h4>
                      <p className="text-sm text-muted-foreground">{viewingPrompt.admin_notes}</p>
                    </div>
                  )}
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

      {/* Submit to Library Dialog */}
      <Dialog open={submissionDialogOpen} onOpenChange={setSubmissionDialogOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Submit Prompt to Library</DialogTitle>
            <DialogDescription>
              Submit your prompt for admin review. Approved prompts will be added to the public library.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label htmlFor="submission-text" className="text-sm font-medium">
                Prompt Text *
              </label>
              <textarea
                id="submission-text"
                className="w-full mt-1 min-h-[200px] rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder="Enter your prompt here..."
                value={submissionText}
                onChange={(e) => setSubmissionText(e.target.value)}
                disabled={isSubmitting}
              />
            </div>
            <div>
              <label htmlFor="submission-notes" className="text-sm font-medium">
                Notes (Optional)
              </label>
              <textarea
                id="submission-notes"
                className="w-full mt-1 min-h-[100px] rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder="Add any notes for the admin (e.g., use cases, context, etc.)"
                value={submissionNotes}
                onChange={(e) => setSubmissionNotes(e.target.value)}
                disabled={isSubmitting}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => setSubmissionDialogOpen(false)}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Button
                onClick={handleSubmitToLibrary}
                disabled={isSubmitting || !submissionText.trim()}
              >
                {isSubmitting ? 'Submitting...' : 'Submit for Review'}
              </Button>
            </div>
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
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-primary dark:text-accent sm:text-3xl">Prompt Library</h1>
                <p className="text-sm text-muted-foreground">
                  A public collection of curated prompts submitted by the community.
                </p>
              </div>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
            <div className="relative w-full sm:w-80">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input 
                placeholder="Search library..."
                aria-label="Search prompt library"
                className="h-11 w-full rounded-full border-transparent bg-muted pl-12 pr-4 transition-colors focus:bg-background focus:ring-2 focus:ring-ring"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            {isAuthenticated && (
              <Button
                onClick={() => setSubmissionDialogOpen(true)}
                className="flex items-center gap-2"
              >
                <Upload className="h-4 w-4" />
                Submit Prompt
              </Button>
            )}
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
                              <div className="flex items-start justify-between">
                                <CardTitle className="flex-1">
                                    {prompt.summary || 'No summary available.'}
                                </CardTitle>
                                {(prompt as ExtendedPrompt).status && (
                                  <div className="ml-2">
                                    {getStatusBadge((prompt as ExtendedPrompt).status)}
                                  </div>
                                )}
                              </div>
                              <CardDescription>
                                Added{' '}
                                {isValidDate(prompt.createdAt)
                                  ? formatDistanceToNow(new Date(prompt.createdAt), {
                                      addSuffix: true,
                                    })
                                  : 'recently'}
                              </CardDescription>
                          </CardHeader>
                          
                          <CardContent className="flex-grow space-y-4 p-4 pt-0 sm:p-6">
                              {prompt.tags && prompt.tags.length > 0 && (
                                  <ul className="flex flex-wrap gap-1.5" aria-label="Prompt tags">
                                      {prompt.tags.map((tag, index) => (
                                          <li key={index}>
                                            <Badge variant="secondary" className="font-normal">
                                                {tag}
                                            </Badge>
                                          </li>
                                      ))}
                                  </ul>
                              )}
                                <p className="text-sm text-foreground/80 line-clamp-3">
                                    {prompt.text}
                                </p>
                          </CardContent>
                          
                          <CardFooter className="flex items-center justify-between p-4 pt-0 sm:p-6">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button 
                                  variant="ghost" 
                                  className="flex items-center gap-1.5 px-2 text-sm text-muted-foreground" 
                                  onClick={() => toggleStar(prompt.id)} 
                                  disabled={!isAuthenticated}
                                  aria-label={prompt.isStarredByUser ? "Un-star this prompt" : "Star this prompt"}
                                >
                                    <Star className={cn("h-4 w-4 transition-colors", prompt.isStarredByUser && "fill-yellow-400 text-yellow-400")} />
                                    {prompt.stars ?? 0}
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>{prompt.isStarredByUser ? "Un-star prompt" : "Star prompt"}</p>
                              </TooltipContent>
                            </Tooltip>
                              
                            <div className='flex items-center gap-2'>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => setViewingPrompt(prompt as ExtendedPrompt)}
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
                                        onClick={() => copyToClipboard(prompt as ExtendedPrompt)}
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

                                {isAdmin && (
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
                                            This action cannot be undone. This will permanently delete this prompt from the public library.
                                        </AlertDialogDescription>
                                        </AlertDialogHeader>
                                        <AlertDialogFooter>
                                        <AlertDialogCancel className={cn(buttonVariants({variant: 'default'}))}>Cancel</AlertDialogCancel>
                                        <AlertDialogAction onClick={() => deleteLibraryPrompt(prompt.id)} className={cn(buttonVariants({variant: 'destructive'}))}>
                                            Delete
                                        </AlertDialogAction>
                                        </AlertDialogFooter>
                                    </AlertDialogContent>
                                    </AlertDialog>
                                )}
                            </div>
                          </CardFooter>
                      </Card>
                  </li>
              ))}
            </ul>
          ) : (
            <div role="region" aria-labelledby="empty-library-heading" className="flex h-[50vh] flex-col items-center justify-center rounded-lg border-2 border-dashed">
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
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
              <h2 id="empty-library-heading" className="text-2xl font-semibold">Library is Empty</h2>
              <p className="mt-2 text-muted-foreground text-center">
                {searchQuery ? "No prompts match your search." : "Be the first to add a prompt to the public library!"}
              </p>
              {isAuthenticated && !searchQuery && (
                <Button
                  onClick={() => setSubmissionDialogOpen(true)}
                  className="mt-4 flex items-center gap-2"
                >
                  <Upload className="h-4 w-4" />
                  Submit First Prompt
                </Button>
              )}
            </div>
          )}
        </ScrollArea>
      </div>
    </TooltipProvider>
  );
}