
'use client';

import * as React from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import type { LibrarySubmission } from '@/hooks/use-prompts';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { format } from 'date-fns';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from './ui/textarea';
import { useToast } from '@/hooks/use-toast';
import { reviewLibrarySubmission } from '@/app/actions';
import { ThumbsUp, ThumbsDown, Eye, FileClock } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from './ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { formatDistanceToNow } from 'date-fns';
import { useAuth } from '@/hooks/use-auth';

type ReviewAction = 'approve' | 'reject';
type Status = 'PENDING' | 'APPROVED' | 'REJECTED';

const isValidDate = (date: any): date is string | number | Date => {
  if (!date) return false;
  const parsedDate = new Date(date);
  return !isNaN(parsedDate.getTime());
};

export function AdminSubmissionsClient({
  initialStatus,
  pendingSubmissions,
  approvedSubmissions,
  rejectedSubmissions,
}: {
  initialStatus: Status;
  pendingSubmissions: LibrarySubmission[];
  approvedSubmissions: LibrarySubmission[];
  rejectedSubmissions: LibrarySubmission[];
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { toast } = useToast();
  const { token } = useAuth();

  const [status, setStatus] = React.useState<Status>(initialStatus);
  const [selectedSubmission, setSelectedSubmission] = React.useState<LibrarySubmission | null>(null);
  const [reviewNotes, setReviewNotes] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);

  const submissionsMap: Record<Status, LibrarySubmission[]> = {
    PENDING: pendingSubmissions,
    APPROVED: approvedSubmissions,
    REJECTED: rejectedSubmissions,
  };

  const currentSubmissions = submissionsMap[status] || [];

  const handleStatusChange = (newStatus: string) => {
    const validStatus = newStatus as Status;
    setStatus(validStatus);
    // Update URL without reloading the page, state change handles the render
    router.push(`${pathname}?status=${validStatus}`, { scroll: false });
  };

  const openReviewDialog = (submission: LibrarySubmission) => {
    setSelectedSubmission(submission);
    setReviewNotes(submission.admin_notes || '');
    setIsDialogOpen(true);
  };

  const handleReview = async (action: ReviewAction) => {
    if (!selectedSubmission || !token) return;

    setIsSubmitting(true);
    try {
      await reviewLibrarySubmission(selectedSubmission.id, action, reviewNotes, token);
      toast({
        title: 'Success',
        description: `Submission has been ${action}d.`,
      });
      setIsDialogOpen(false);
      // This will force a server-side refetch of the page with new props
      router.refresh(); 
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message || 'Failed to process review.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  React.useEffect(() => {
    setStatus(initialStatus);
  }, [initialStatus]);


  return (
    <TooltipProvider>
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Review Prompt Submission</DialogTitle>
            <DialogDescription>Review the prompt and approve or reject it.</DialogDescription>
          </DialogHeader>
          {selectedSubmission && (
            <div className="space-y-4 py-4">
              <div className="max-h-60 overflow-y-auto rounded-md border bg-muted p-3">
                <p className="text-sm whitespace-pre-wrap">{selectedSubmission.prompt_text}</p>
              </div>
              <div className="text-sm">
                <p>
                  <span className="font-semibold">Submitted by:</span> {selectedSubmission.user?.full_name || 'N/A'} ({selectedSubmission.user?.email || 'Unknown User'})
                </p>
                <p>
                  <span className="font-semibold">User Notes:</span> {selectedSubmission.submission_notes || 'N/A'}
                </p>
              </div>
              <Textarea
                placeholder="Add optional admin notes..."
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
              />
            </div>
          )}
          <DialogFooter>
            <Button variant="secondary" onClick={() => setIsDialogOpen(false)}>Cancel</Button>
            <Button
              variant="destructive"
              onClick={() => handleReview('reject')}
              disabled={isSubmitting}
            >
              <ThumbsDown className="mr-2 h-4 w-4" /> Reject
            </Button>
            <Button 
                onClick={() => handleReview('approve')} 
                disabled={isSubmitting}
                className="bg-green-600 hover:bg-green-700 text-white"
            >
              <ThumbsUp className="mr-2 h-4 w-4" /> Approve
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Card>
        <CardHeader>
          <Tabs value={status} onValueChange={handleStatusChange}>
            <TabsList>
              <TabsTrigger value="PENDING">Pending ({pendingSubmissions.length})</TabsTrigger>
              <TabsTrigger value="APPROVED">Approved ({approvedSubmissions.length})</TabsTrigger>
              <TabsTrigger value="REJECTED">Rejected ({rejectedSubmissions.length})</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {currentSubmissions.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead className="hidden sm:table-cell">Submitted</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {currentSubmissions.map((sub) => (
                  <TableRow key={sub.id}>
                    <TableCell>
                      <div className="font-medium">{sub.user?.full_name || 'N/A'}</div>
                      <div className="text-sm text-muted-foreground">{sub.user?.email}</div>
                    </TableCell>
                    <TableCell className="hidden sm:table-cell">
                      {isValidDate(sub.submitted_at) ? (
                        <Tooltip>
                          <TooltipTrigger>
                            <span className="cursor-default">{format(new Date(sub.submitted_at), 'MMM d, yyyy')}</span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{formatDistanceToNow(new Date(sub.submitted_at), { addSuffix: true })}</p>
                          </TooltipContent>
                        </Tooltip>
                      ) : (
                        'Invalid date'
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          sub.status === 'APPROVED' ? 'default' : sub.status === 'REJECTED' ? 'destructive' : 'secondary'
                        }
                        className={sub.status === 'APPROVED' ? "bg-green-600" : ""}
                      >
                        {sub.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="icon" onClick={() => openReviewDialog(sub)}>
                        <Eye className="h-4 w-4" />
                        <span className="sr-only">Review</span>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center p-8 text-center">
              <FileClock className="h-12 w-12 text-muted-foreground" />
              <p className="mt-4 font-semibold">No submissions found</p>
              <p className="text-sm text-muted-foreground">There are no submissions with the status "{status}".</p>
            </div>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}
