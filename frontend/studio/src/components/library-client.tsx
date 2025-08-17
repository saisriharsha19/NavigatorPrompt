
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
import { Clipboard, Check, Search, Star, Trash2, Eye, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { useAuth } from '@/hooks/use-auth';
import { cn } from '@/lib/utils';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { formatDistanceToNow } from 'date-fns';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { useIsMobile } from '@/hooks/use-mobile';
import { motion, AnimatePresence } from 'framer-motion';

// Extended Prompt type to include submission data
interface ExtendedPrompt extends Prompt {
  status?: 'APPROVED' | 'PENDING' | 'REJECTED';
  admin_notes?: string;
  submission_notes?: string;
}

function PromptRowSkeleton() {
  return (
    <TableRow>
      <TableCell className="w-1/2">
        <Skeleton className="h-5 w-3/4" />
        <Skeleton className="h-4 w-1/4 mt-2" />
      </TableCell>
      <TableCell className="hidden sm:table-cell">
        <div className="flex flex-wrap gap-1.5">
            <Skeleton className="h-5 w-14 rounded-full" />
            <Skeleton className="h-5 w-20 rounded-full" />
        </div>
      </TableCell>
      <TableCell className="hidden md:table-cell">
        <Skeleton className="h-5 w-16" />
      </TableCell>
       <TableCell className="hidden md:table-cell">
        <Skeleton className="h-5 w-8" />
      </TableCell>
      <TableCell className="text-right">
        <div className="flex items-center justify-end gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
      </TableCell>
    </TableRow>
  );
}

function PromptCardSkeleton() {
    return (
      <Card className="flex flex-col">
        <CardHeader>
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-4 w-1/3 mt-2" />
        </CardHeader>
        <CardContent className="flex-grow space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
        </CardContent>
        <CardFooter className="flex justify-between items-center">
            <div className='flex items-center gap-2'>
                <Skeleton className="h-5 w-14" />
            </div>
            <div className="flex justify-end gap-2">
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
        <Badge variant="default" className="flex items-center gap-1 bg-green-600/20 text-green-400 border-green-600/30 hover:bg-green-600/30">
          <CheckCircle className="h-3 w-3" />
          Approved
        </Badge>
      );
    case 'PENDING':
      return (
        <Badge variant="secondary" className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          Pending
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
  const { libraryPrompts, toggleStar, isLoading, deleteLibraryPrompt } = useLibrary();
  const { isAuthenticated, isAdmin } = useAuth();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewingPrompt, setViewingPrompt] = useState<ExtendedPrompt | null>(null);
  const { toast } = useToast();
  const isMobile = useIsMobile();

  const copyToClipboard = async (prompt: ExtendedPrompt) => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(prompt.text);
      } else {
        // Fallback for browsers without clipboard API
        const textArea = document.createElement('textarea');
        textArea.value = prompt.text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
      
      setCopiedId(prompt.id);
      toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
      toast({ 
        variant: 'destructive',
        title: 'Copy failed', 
        description: 'Could not copy to clipboard.' 
      });
    }
  };
  
  const filteredPrompts = libraryPrompts.filter(prompt => {
    const query = searchQuery.toLowerCase();
    const textMatch = prompt.text.toLowerCase().includes(query);
    const summaryMatch = prompt.summary?.toLowerCase().includes(query);
    const tagsMatch = prompt.tags?.some(tag => tag.toLowerCase().includes(query));
    return textMatch || summaryMatch || tagsMatch;
  });
  const isValidDate = (date: any) => {
      if (!date) return false;
      const parsed = new Date(date);
      return !isNaN(parsed.getTime()) && parsed.getFullYear() > 1970;
  };

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

  const renderDesktopView = () => (
    <Card>
        <CardContent className="p-0">
            <Table>
                <TableHeader>
                    <TableRow>
                        <TableHead className="w-2/5 pl-6">Prompt</TableHead>
                        <TableHead>Tags</TableHead>
                        <TableHead>Added</TableHead>
                        <TableHead className="text-center">Stars</TableHead>
                        <TableHead className="text-right pr-6">Actions</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    {isLoading ? (
                       Array.from({ length: 5 }).map((_, i) => <PromptRowSkeleton key={i} />)
                    ) : filteredPrompts.length > 0 ? (
                        filteredPrompts.map((prompt) => (
                            <TableRow key={prompt.id}>
                                <TableCell className="pl-6">
                                    <p className="font-medium truncate">{prompt.summary || 'No summary available.'}</p>
                                    <p className="text-sm text-muted-foreground line-clamp-1">{prompt.text}</p>
                                </TableCell>
                                <TableCell>
                                    {prompt.tags && prompt.tags.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5" aria-label="Prompt tags">
                                            {prompt.tags.slice(0, 2).map((tag, index) => (
                                                <Badge key={index} variant="secondary" className="font-normal">
                                                    {tag}
                                                </Badge>
                                            ))}
                                            {prompt.tags.length > 2 && <Badge variant="outline">+{prompt.tags.length - 2}</Badge>}
                                        </div>
                                    )}
                                </TableCell>
                                <TableCell>
                                    <span className="text-muted-foreground text-sm">
                                        {isValidDate(prompt.created_at)
                                        ? formatDistanceToNow(new Date(prompt.created_at), { addSuffix: true })
                                        : 'recently'}
                                    </span>
                                </TableCell>
                                <TableCell className="text-center">
                                     <span className="font-medium">{prompt.stars ?? 0}</span>
                                </TableCell>
                                <TableCell className="pr-6">
                                    <div className='flex items-center justify-end gap-1'>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <Button 
                                                variant="ghost" 
                                                size="icon"
                                                className="h-8 w-8"
                                                onClick={() => toggleStar(prompt.id)} 
                                                disabled={!isAuthenticated}
                                                aria-label={prompt.isStarredByUser ? "Un-star this prompt" : "Star this prompt"}
                                                >
                                                    <Star className={cn("h-4 w-4 transition-colors", prompt.isStarredByUser && "fill-yellow-400 text-yellow-400")} />
                                                </Button>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>{prompt.isStarredByUser ? "Un-star prompt" : "Star prompt"}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-8 w-8"
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
                                                className="h-8 w-8"
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
                                                    <Button variant="ghost" size="icon" className="text-destructive/70 hover:bg-destructive/10 hover:text-destructive focus-visible:ring-destructive h-8 w-8" aria-label="Delete prompt">
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
                                                <AlertDialogCancel className={cn(buttonVariants({variant: 'outline'}))}>Cancel</AlertDialogCancel>
                                                <AlertDialogAction onClick={() => deleteLibraryPrompt(prompt.id)} className={cn(buttonVariants({variant: 'destructive'}))}>
                                                    Delete
                                                </AlertDialogAction>
                                                </AlertDialogFooter>
                                            </AlertDialogContent>
                                            </AlertDialog>
                                        )}
                                    </div>
                                </TableCell>
                            </TableRow>
                        ))
                    ) : (
                        <TableRow>
                            <TableCell colSpan={5} className="h-48 text-center">
                                {renderEmptyState()}
                            </TableCell>
                        </TableRow>
                    )}
                </TableBody>
            </Table>
        </CardContent>
    </Card>
  );

  const renderMobileView = () => (
    <div className="space-y-4">
        {isLoading ? (
            Array.from({ length: 5 }).map((_, i) => <PromptCardSkeleton key={i} />)
        ) : filteredPrompts.length > 0 ? (
            <motion.div
                className="space-y-4"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
            <AnimatePresence>
            {filteredPrompts.map(prompt => (
                <motion.div key={prompt.id} variants={itemVariants} layout>
                <Card>
                    <CardHeader>
                        <CardTitle className="text-base line-clamp-2">{prompt.summary || 'No summary'}</CardTitle>
                          <CardDescription>
                              Added {isValidDate(prompt.created_at) ? formatDistanceToNow(new Date(prompt.created_at), { addSuffix: true }) : 'recently'}
                          </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {prompt.tags && prompt.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-4" aria-label="Prompt tags">
                                {prompt.tags.map((tag, index) => (
                                    <Badge key={index} variant="secondary" className="font-normal">
                                        {tag}
                                    </Badge>
                                ))}
                            </div>
                        )}
                        <p className="text-sm text-muted-foreground line-clamp-3">{prompt.text}</p>
                    </CardContent>
                    <CardFooter className="flex justify-between items-center">
                        <Button 
                            variant="outline" 
                            size="sm"
                            className="flex items-center gap-1.5 px-2 text-sm text-muted-foreground" 
                            onClick={() => toggleStar(prompt.id)} 
                            disabled={!isAuthenticated}
                            aria-label={prompt.isStarredByUser ? "Un-star this prompt" : "Star this prompt"}
                            >
                                <Star className={cn("h-4 w-4 transition-colors", prompt.isStarredByUser && "fill-yellow-400 text-yellow-400")} />
                                <span>{prompt.stars ?? 0}</span>
                        </Button>
                        <div className='flex items-center justify-end gap-1'>
                             <Button
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => setViewingPrompt(prompt as ExtendedPrompt)}
                                aria-label="View prompt"
                            >
                                <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => copyToClipboard(prompt as ExtendedPrompt)}
                                aria-label="Copy prompt"
                            >
                                {copiedId === prompt.id ? (
                                <Check className="h-4 w-4 text-primary" />
                                ) : (
                                <Clipboard className="h-4 w-4" />
                                )}
                            </Button>
                            {isAdmin && (
                                <AlertDialog>
                                <AlertDialogTrigger asChild>
                                    <Button variant="ghost" size="icon" className="text-destructive/70 hover:bg-destructive/10 hover:text-destructive focus-visible:ring-destructive h-9 w-9" aria-label="Delete prompt">
                                        <Trash2 className="h-4 w-4 transition-colors" />
                                    </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                    <AlertDialogHeader>
                                        <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                        <AlertDialogCancel>Cancel</AlertDialogCancel>
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
                </motion.div>
            ))}
            </AnimatePresence>
            </motion.div>
        ) : (
            <div className="h-48 flex items-center justify-center">
                {renderEmptyState()}
            </div>
        )}
    </div>
  );

  const renderEmptyState = () => (
    <div role="region" aria-labelledby="empty-library-heading" className="text-center">
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="mx-auto mb-4 h-12 w-12 text-muted-foreground"
            aria-hidden="true"
        >
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
        </svg>
        <h2 id="empty-library-heading" className="text-xl font-semibold">Library is Empty</h2>
        <p className="mt-2 text-muted-foreground">
            {searchQuery ? "No prompts match your search." : "Go to the Generator to submit a prompt to the library!"}
        </p>
    </div>
  );


  return (
    <TooltipProvider>
      {/* View Prompt Dialog */}
      <Dialog open={!!viewingPrompt} onOpenChange={(isOpen) => !isOpen && setViewingPrompt(null)}>
        <DialogContent className="sm:max-w-2xl w-[95vw] max-h-[85vh] flex flex-col p-0">
            <DialogHeader className="p-6 pb-0">
                <div className="flex items-center justify-between">
                <DialogTitle className="truncate">{viewingPrompt?.summary || 'Prompt Details'}</DialogTitle>
                {viewingPrompt?.status && getStatusBadge(viewingPrompt.status)}
                </div>
            </DialogHeader>
            <div className="relative flex-1 min-h-0 overflow-hidden px-6">
                <ScrollArea className="h-full pr-6 -mr-6">
                    <div className="space-y-6 pb-6">
                        {/* Tags */}
                        {viewingPrompt?.tags && viewingPrompt.tags.length > 0 && (
                        <div className="space-y-2">
                            <h4 className="font-medium">Tags</h4>
                            <div className="flex flex-wrap gap-2">
                            {viewingPrompt.tags.map((tag) => (
                                <Badge key={tag} variant="secondary">{tag}</Badge>
                            ))}
                            </div>
                        </div>
                        )}

                        {/* Prompt Text */}
                        <div className="space-y-2">
                        <h4 className="font-medium">Prompt Text</h4>
                        <div className="p-4 rounded-md border bg-muted/50">
                          <ScrollArea className="h-64 w-full">
                              <p className="text-sm text-foreground whitespace-pre-wrap break-words pr-4">
                              {viewingPrompt?.text}
                              </p>
                          </ScrollArea>
                        </div>
                        </div>

                        {/* Submitter Notes */}
                        {viewingPrompt?.submission_notes && (
                        <div className="space-y-2">
                            <h4 className="font-medium">Submitter Notes</h4>
                            <p className="text-sm text-muted-foreground italic">
                            "{viewingPrompt.submission_notes}"
                            </p>
                        </div>
                        )}

                        {/* Admin Notes */}
                        {viewingPrompt?.admin_notes && (
                        <div className="space-y-2">
                            <h4 className="font-medium">Admin Notes</h4>
                            <p className="text-sm text-muted-foreground italic">
                            "{viewingPrompt.admin_notes}"
                            </p>
                        </div>
                        )}
                    </div>
                </ScrollArea>
            </div>
            {viewingPrompt && (
                <div className="p-6 pt-0">
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="outline"
                                className="w-full"
                                onClick={() => copyToClipboard(viewingPrompt)}
                                aria-label="Copy prompt"
                            >
                                {copiedId === viewingPrompt.id ? <Check /> : <Clipboard />}
                                <span>{copiedId === viewingPrompt.id ? 'Copied!' : 'Copy Prompt to Clipboard'}</span>
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            <p>Copy prompt</p>
                        </TooltipContent>
                    </Tooltip>
                </div>
            )}
        </DialogContent>
      </Dialog>

      <div className="container mx-auto max-w-7xl py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex-1">
            <h1 className="text-2xl font-bold tracking-tight text-primary sm:text-3xl">Prompt Library</h1>
            <p className="text-sm text-muted-foreground mt-1">
              A public collection of curated prompts submitted by the community.
            </p>
          </div>
          <div className="flex w-full sm:w-auto">
            <div className="relative w-full sm:w-72">
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
        </div>
        
        {isMobile === undefined ? (
            <div className='space-y-4'>
                <PromptCardSkeleton />
                <PromptCardSkeleton />
                <PromptCardSkeleton />
            </div>
        ) : isMobile ? renderMobileView() : renderDesktopView()}

      </div>
    </TooltipProvider>
  );
}

    