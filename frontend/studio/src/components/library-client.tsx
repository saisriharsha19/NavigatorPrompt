
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
import { Clipboard, Check, Search, Star, Trash2, Eye } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { useAuth } from '@/hooks/use-auth';
import { cn } from '@/lib/utils';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { formatDistanceToNow } from 'date-fns';
import { motion } from 'framer-motion';

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

const containerVariants = {
  hidden: { opacity: 1 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
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


export function LibraryClient() {
  const { libraryPrompts, toggleStar, isLoading, deleteLibraryPrompt } = useLibrary();
  const { isAuthenticated, isAdmin } = useAuth();
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
  
  const filteredPrompts = libraryPrompts.filter(prompt => {
    const query = searchQuery.toLowerCase();
    const textMatch = prompt.text.toLowerCase().includes(query);
    const summaryMatch = prompt.summary?.toLowerCase().includes(query);
    const tagsMatch = prompt.tags?.some(tag => tag.toLowerCase().includes(query));
    return textMatch || summaryMatch || tagsMatch;
  });
  
  const isValidDate = (date: any) => date && !isNaN(new Date(date).getTime());

  return (
    <TooltipProvider>
      <Dialog open={!!viewingPrompt} onOpenChange={(isOpen) => !isOpen && setViewingPrompt(null)}>
          <DialogContent className="sm:max-w-2xl">
              <DialogHeader>
                  <DialogTitle>{viewingPrompt?.summary || 'Prompt Details'}</DialogTitle>
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
        </div>

        <ScrollArea className="h-full">
          {isLoading ? (
            <ul className="grid grid-cols-1 gap-6">
              {Array.from({ length: 8 }).map((_, i) => <li key={i}><PromptCardSkeleton /></li>)}
            </ul>
          ) : filteredPrompts.length > 0 ? (
            <motion.ul
              className="grid grid-cols-1 gap-6"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              {filteredPrompts.map((prompt) => (
                  <motion.li key={prompt.id} variants={itemVariants}>
                      <Card className="flex h-full flex-col">
                          <CardHeader className="p-4 sm:p-6">
                              <CardTitle>
                                  {prompt.summary || 'No summary available.'}
                              </CardTitle>
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
                  </motion.li>
              ))}
            </motion.ul>
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
              <p className="mt-2 text-muted-foreground">
                {searchQuery ? "No prompts match your search." : "Be the first to add a prompt to the public library!"}
              </p>
            </div>
          )}
        </ScrollArea>
      </div>
    </TooltipProvider>
  );
}
