
'use client';

import { useState } from 'react';
import type { Prompt } from '@/hooks/use-prompts';
import { useLibrary } from '@/hooks/use-library';
import { Button } from '@/components/ui/button';
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
import { ScrollArea } from '@/components/ui/scroll-area';
import { Clipboard, Check, Trash2, Library, Search, UserCircle, Lock } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { formatDistanceToNow } from 'date-fns';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { useAuth } from '@/hooks/use-auth';

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

export function PromptLibraryClient() {
  const { libraryPrompts, deleteLibraryPrompt, isLoading } = useLibrary();
  const { isAuthenticated, login } = useAuth();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const { toast } = useToast();

  const copyToClipboard = (prompt: Prompt) => {
    navigator.clipboard.writeText(prompt.text);
    setCopiedId(prompt.id);
    toast({ title: 'Copied!', description: 'Prompt copied to clipboard.' });
    setTimeout(() => setCopiedId(null), 2000);
  };
  
  const filteredPrompts = libraryPrompts.filter(prompt => 
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
            <h2 className="mt-6 text-2xl font-semibold tracking-tight">Access Your Private Library</h2>
            <p className="mt-2 text-muted-foreground">
              Sign in to view, save, and manage your curated collection of prompts.
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
    <div className="container mx-auto max-w-7xl py-8 px-4 sm:px-6 lg:px-8">
      <div className="mb-8 flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className='flex items-center gap-3'>
          <Library className="h-8 w-8" />
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Prompt Library</h1>
            <p className="text-muted-foreground">
              Your collection of curated prompts. You have saved {libraryPrompts.length} of 50 prompts.
            </p>
          </div>
        </div>
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <Input 
            placeholder="Search library..."
            className="h-11 w-full rounded-full border-transparent bg-muted pl-12 pr-4 transition-colors focus:bg-background focus:ring-2 focus:ring-ring"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <ScrollArea className="h-full">
        {isLoading ? (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => <PromptCardSkeleton key={i} />)}
          </div>
        ) : filteredPrompts.length > 0 ? (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredPrompts.map((prompt) => (
              <Card key={prompt.id} className="flex flex-col">
                <CardHeader>
                  <CardTitle className="text-lg leading-snug">
                    {getPromptTitle(prompt.text)}
                  </CardTitle>
                  <CardDescription>
                    Saved{' '}
                    {formatDistanceToNow(new Date(prompt.createdAt), {
                      addSuffix: true,
                    })}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <p className="line-clamp-6 text-sm text-foreground/80">
                    {prompt.text}
                  </p>
                </CardContent>
                <CardFooter className="flex justify-end gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => copyToClipboard(prompt)}
                  >
                    {copiedId === prompt.id ? (
                      <Check className="h-4 w-4 text-primary" />
                    ) : (
                      <Clipboard className="h-4 w-4" />
                    )}
                    <span className="sr-only">Copy</span>
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <Trash2 className="h-4 w-4 text-destructive" />
                        <span className="sr-only">Delete</span>
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This action cannot be undone. This will permanently delete this prompt from your library.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={() => deleteLibraryPrompt(prompt.id)}>
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </CardFooter>
              </Card>
            ))}
          </div>
        ) : (
          <div className="flex h-[50vh] flex-col items-center justify-center rounded-lg border-2 border-dashed">
            <Library className="mb-4 h-16 w-16 text-muted-foreground" />
            <h2 className="text-2xl font-semibold">Library is Empty</h2>
            <p className="mt-2 text-muted-foreground">
              {searchQuery ? "No prompts match your search." : "Save a finalized prompt from the generator to start your collection."}
            </p>
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
