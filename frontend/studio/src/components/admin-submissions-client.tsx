
'use client';

import * as React from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { LibrarySubmission } from '@/hooks/use-prompts';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatDistanceToNow } from 'date-fns';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Textarea } from './ui/textarea';
import { useToast } from '@/hooks/use-toast';
import { reviewLibrarySubmission } from '@/app/actions';
import { ThumbsUp, ThumbsDown, Eye, FileClock, User } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from './ui/tabs';

type ReviewAction = 'approve' | 'reject';

export function AdminSubmissionsClient({
  initialSubmissions,
  initialStatus,
}: {
  initialSubmissions: LibrarySubmission[];
  initialStatus: 'PENDING' | 'APPROVED' | 'REJECTED';
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { toast } = useToast();

  const [submissions, setSubmissions] = React.useState(initialSubmissions);
  const [selectedSubmission, setSelectedSubmission] = React.useState<LibrarySubmission | null>(null);
  const [reviewNotes, setReviewNotes] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);

  React.useEffect(() => {
    setSubmissions(initialSubmissions);
  }, [initialSubmissions]);

  const handleStatusChange = (status: string) => {
    const params = new URLSearchParams(window.location.search);
    params.set('status', status);
    router.replace(`${pathname}?${params.toString()}`);
  };

  const openReviewDialog = (submission: LibrarySubmission) => {
    setSelectedSubmission(submission);
    setReviewNotes('');
    setIsDialogOpen(true);
  };

  const handleReview = async (action: ReviewAction) => {
    if (!selectedSubmission) return;

    setIsSubmitting(true);
    try {
      await reviewLibrarySubmission(selectedSubmission.id, action, reviewNotes);
      toast({
        title: 'Success',
        description: `Submission has been ${action}d.`,
      });
      // Refresh data
      handleStatusChange(initialStatus);
      setIsDialogOpen(false);
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

  return (
    <>
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
                  <span className="font-semibold">Submitted by:</span> {selectedSubmission.user?.email || 'Unknown User'}
                </p>
                <p>
                  <span className="font-semibold">Notes:</span> {selectedSubmission.submission_notes || 'N/A'}
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
              <ThumbsDown className="mr-2" /> Reject
            </Button>
            <Button 
                onClick={() => handleReview('approve')} 
                disabled={isSubmitting}
                className="bg-green-600 hover:bg-green-700 text-white"
            >
              <ThumbsUp className="mr-2" /> Approve
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Card>
        <CardHeader>
          <Tabs defaultValue={initialStatus} onValueChange={handleStatusChange}>
            <TabsList>
              <TabsTrigger value="PENDING">Pending</TabsTrigger>
              <TabsTrigger value="APPROVED">Approved</TabsTrigger>
              <TabsTrigger value="REJECTED">Rejected</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {submissions.length > 0 ? (
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
                {submissions.map((sub) => (
                  <TableRow key={sub.id}>
                    <TableCell>
                      <div className="font-medium">{sub.user?.full_name || 'N/A'}</div>
                      <div className="text-sm text-muted-foreground">{sub.user?.email}</div>
                    </TableCell>
                    <TableCell className="hidden sm:table-cell">
                      {formatDistanceToNow(new Date(sub.submitted_at), { addSuffix: true })}
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
              <p className="text-sm text-muted-foreground">There are no submissions with the status "{initialStatus}".</p>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
