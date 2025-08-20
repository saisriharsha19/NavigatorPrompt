
'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileText, X, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { useAuth } from '@/hooks/use-auth';
import { Button } from './ui/button';
import { usePromptForge } from '@/hooks/use-prompt-forge';
import mammoth from 'mammoth';
import * as pdfjsLib from 'pdfjs-dist';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

pdfjsLib.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ALLOWED_FILE_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
  'text/plain',
  'text/markdown',
];

export function DocumentManager() {
  const { user } = useAuth();
  const { uploadedFiles, setUploadedFiles } = usePromptForge();
  const [isExtracting, setIsExtracting] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();

  const handleFileExtraction = async (file: File) => {
    // --- Validation ---
    if (file.size > MAX_FILE_SIZE_BYTES) {
      toast({
        variant: 'destructive',
        title: 'File Too Large',
        description: `${file.name} is over the ${MAX_FILE_SIZE_MB}MB limit.`,
      });
      return null;
    }
    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      toast({
        variant: 'destructive',
        title: 'Invalid File Type',
        description: `File type for ${file.name} is not supported.`,
      });
      return null;
    }
    // --- End Validation ---

    try {
      let textContent = '';
      if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        textContent = result.value;
      } else if (file.type === 'application/pdf') {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
        const numPages = pdf.numPages;
        const textParts = [];
        for (let i = 1; i <= numPages; i++) {
          const page = await pdf.getPage(i);
          const pageTextContent = await page.getTextContent();
          const pageText = pageTextContent.items.map(item => ('str' in item ? item.str : '')).join(' ');
          textParts.push(pageText);
        }
        textContent = textParts.join('\n\n');
      } else {
        textContent = await file.text();
      }

      // Sanitize the text content to make it JSON-safe.
      // 1. Remove non-printable control characters.
      const cleanedText = textContent.replace(/[\u0000-\u001F\u007F-\u009F]/g, "");
      // 2. Stringify and slice to escape all quotes, backslashes, etc.
      const jsonSafeContent = JSON.stringify(cleanedText).slice(1, -1);

      return {
        id: `${file.name}-${file.lastModified}`,
        name: file.name,
        content: jsonSafeContent,
      };
    } catch (error: any) {
      console.error(`File extraction error for ${file.name}:`, error);
      toast({ variant: 'destructive', title: `Extraction Failed for ${file.name}`, description: 'Could not read text from the document.' });
      return null;
    }
  };

  const handleFilesChange = async (selectedFiles: FileList | null) => {
    if (!user) {
      toast({ variant: 'destructive', title: 'Please sign in to use documents.' });
      return;
    }
    if (!selectedFiles || selectedFiles.length === 0) return;

    setIsExtracting(true);

    const extractionPromises = Array.from(selectedFiles).map(handleFileExtraction);
    const newFiles = (await Promise.all(extractionPromises)).filter(Boolean) as { id: string, name: string, content: string }[];
    
    if (newFiles.length > 0) {
      setUploadedFiles(prev => {
        const existingIds = new Set(prev.map(f => f.id));
        const trulyNewFiles = newFiles.filter(f => !existingIds.has(f.id));
        return [...prev, ...trulyNewFiles];
      });
      toast({ title: 'Extraction Successful', description: `Added ${newFiles.length} document(s) to context.` });
    }

    setIsExtracting(false);
  };

  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };
  
  const handleDrop = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
    if (event.dataTransfer.files) {
      handleFilesChange(event.dataTransfer.files);
    }
  }, [user, handleFilesChange]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
  }, []);

  return (
    <div className="space-y-6 rounded-lg border bg-card p-6 text-card-foreground shadow-sm">
      <div className="space-y-1.5">
        <h3 className="text-lg font-medium leading-snug">Context & Knowledge</h3>
        <p className="text-sm text-muted-foreground">Upload documents to provide context for the assistant.</p>
      </div>

      <AnimatePresence mode="wait">
        {isExtracting ? (
            <motion.div
              key="loading-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-12 text-center"
          >
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="font-semibold">Extracting text...</p>
              <p className="text-xs text-muted-foreground">Please wait while documents are processed.</p>
          </motion.div>
        ) : (
          <motion.div
            key="upload-view"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <label
              htmlFor="file-upload"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={cn(
                'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-12 text-center transition-colors',
                isDragOver ? 'border-primary bg-accent dark:border-accent' : 'border-border hover:border-primary/50 dark:hover:border-accent',
                !user && 'cursor-not-allowed opacity-60'
              )}
            >
              <div className="rounded-full bg-muted p-3">
                  <UploadCloud className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="font-semibold">
                <span className="text-primary dark:text-accent">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, or MD (max. 10MB each)</p>
            </label>
            <input
              id="file-upload"
              type="file"
              className="sr-only"
              onChange={(e) => handleFilesChange(e.target.files)}
              accept={ALLOWED_FILE_TYPES.join(',')}
              disabled={!user || isExtracting}
              multiple
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium">Active Documents</h4>
          {uploadedFiles.length > 0 && (
            <span className="text-xs text-green-600 dark:text-green-400 font-medium">
              ✓ {uploadedFiles.length} document{uploadedFiles.length !== 1 ? 's' : ''} will be included
            </span>
          )}
        </div>
        <div className="min-h-[64px] rounded-lg border bg-muted/50 p-1">
          <div className="max-h-48 space-y-1 overflow-y-auto p-1">
            <TooltipProvider>
              {uploadedFiles.length > 0 ? (
                <ul className="space-y-1">
                  {uploadedFiles.map((file, index) => (
                    <li key={file.id} className="flex items-center gap-2 rounded-md bg-background p-2 shadow-sm animate-in fade-in-50">
                      <FileText className="h-5 w-5 shrink-0 text-primary dark:text-accent" />
                      <Tooltip delayDuration={300}>
                        <TooltipTrigger asChild>
                          <div className="flex-1 min-w-0">
                            <span className="text-sm font-medium text-foreground truncate block">
                              {file.name}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {file.content.length} characters extracted
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="max-w-xs">
                            <p className="font-medium">{file.name}</p>
                            <p className="text-xs mt-1">
                              {file.content.substring(0, 100)}...
                            </p>
                          </div>
                        </TooltipContent>
                      </Tooltip>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 shrink-0"
                        onClick={() => handleRemoveFile(file.id)}
                        disabled={isExtracting}
                        aria-label={`Remove ${file.name}`}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="flex h-16 items-center justify-center">
                  <p className="text-sm text-muted-foreground">
                    {!user ? 'Sign in to add documents.' : 'No document context applied.'}
                  </p>
                </div>
              )}
            </TooltipProvider>
          </div>
        </div>
        {uploadedFiles.length > 0 && (
          <div className="text-xs text-muted-foreground bg-blue-50 dark:bg-blue-950 p-2 rounded border-l-2 border-blue-500">
            <strong>Note:</strong> These documents will be automatically included as context in all AI operations (generate, evaluate, iterate).
          </div>
        )}
      </div>
    </div>
  );
}
