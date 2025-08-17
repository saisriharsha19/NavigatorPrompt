'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { CheckCircle, XCircle, Clock, Eye } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { formatDistanceToNow } from 'date-fns';
import { LibrarySubmission, reviewLibrarySubmission } from '@/app/actions';
import Cookies from 'js-cookie';

type AdminSubmissionsClientProps = {
  initialStatus: 'PENDING' | 'APPROVED' | 'REJECTED';
  pendingSubmissions: LibrarySubmission[];
  approvedSubmissions: LibrarySubmission[];
  rejectedSubmissions: LibrarySubmission[];
};

export function AdminSubmissionsClient({
  initialStatus,
  pendingSubmissions,
  approvedSubmissions,
  rejectedSubmissions,
}: AdminSubmissionsClientProps) {
  const [activeTab, setActiveTab] = useState<'PENDING' | 'APPROVED' | 'REJECTED'>(initialStatus);
  const [viewingSubmission, setViewingSubmission] = useState<LibrarySubmission | null>(null);
  const [reviewingSubmission, setReviewingSubmission] = useState<LibrarySubmission | null>(null);
  const [reviewAction, setReviewAction] = useState<'approve' | 'reject' | null>(null);
  const [adminNotes, setAdminNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleTabChange = (value: string) => {
    setActiveTab(value as 'PENDING' | 'APPROVED' | 'REJECTED');
  };

  let currentTabSubmissions: LibrarySubmission[] = [];
  if (activeTab === 'PENDING') currentTabSubmissions = pendingSubmissions;
  if (activeTab === 'APPROVED') currentTabSubmissions = approvedSubmissions;
  if (activeTab === 'REJECTED') currentTabSubmissions = rejectedSubmissions;

  function getStatusIcon(status: string): import("react").ReactNode {
    switch (status) {
      case 'APPROVED':
        return <CheckCircle className="text-green-500 h-5 w-5" />;
      case 'REJECTED':
        return <XCircle className="text-red-500 h-5 w-5" />;
      case 'PENDING':
      default:
        return <Clock className="text-yellow-500 h-5 w-5" />;
    }
  }
  function getStatusBadge(status: string): import("react").ReactNode {
    switch (status) {
      case 'APPROVED':
        return <Badge className="bg-green-100 text-green-700 border-green-200">Approved</Badge>;
      case 'REJECTED':
        return <Badge className="bg-red-100 text-red-700 border-red-200">Rejected</Badge>;
      case 'PENDING':
      default:
        return <Badge className="bg-yellow-100 text-yellow-700 border-yellow-200">Pending</Badge>;
    }
  }

  // Approve/Reject handler
  async function handleReview() {
    if (!reviewingSubmission || !reviewAction) return;
    setIsSubmitting(true);
    try {
      const token = Cookies.get('auth_token');
      await reviewLibrarySubmission(
        reviewingSubmission.id,
        reviewAction,
        adminNotes,
        token
      );
      // Optionally, reload or update submissions here
      setReviewingSubmission(null);
      setReviewAction(null);
      setAdminNotes('');
      // You may want to trigger a refresh from parent or refetch here
      window.location.reload(); // quick way to refresh for demo
    } catch (error) {
      alert('Failed to review submission.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="flex flex-row w-full">
          <TabsTrigger value="PENDING" className="flex-1 text-xs sm:text-base truncate">
            Pending ({pendingSubmissions.length})
          </TabsTrigger>
          <TabsTrigger value="APPROVED" className="flex-1 text-xs sm:text-base truncate">
            Approved ({approvedSubmissions.length})
          </TabsTrigger>
          <TabsTrigger value="REJECTED" className="flex-1 text-xs sm:text-base truncate">
            Rejected ({rejectedSubmissions.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="space-y-2 sm:space-y-4">
          {currentTabSubmissions.length === 0 ? (
            <Card className="w-full">
              <CardContent className="flex items-center justify-center py-6 sm:py-8">
                <p className="text-muted-foreground text-sm sm:text-base">
                  No {activeTab.toLowerCase()} submissions found.
                </p>
              </CardContent>
            </Card>
          ) : (
            currentTabSubmissions.map((submission) => (
              <Card key={submission.id} className="w-full">
                <CardHeader>
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                    <div className="space-y-1">
                      <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                        {getStatusIcon(submission.status)}
                        Submission from {submission.user_email || 'Unknown User'}
                      </CardTitle>
                      <CardDescription className="text-xs sm:text-sm">
                        Submitted {formatDistanceToNow(new Date(submission.created_at), { addSuffix: true })}
                        {submission.reviewed_at && (
                          <> • Reviewed {formatDistanceToNow(new Date(submission.reviewed_at), { addSuffix: true })}</>
                        )}
                      </CardDescription>
                    </div>
                    {getStatusBadge(submission.status)}
                  </div>
                </CardHeader>
                <CardContent className="space-y-2 sm:space-y-4">
                  <div>
                    <h4 className="font-medium mb-1 sm:mb-2 text-sm sm:text-base">Prompt Text:</h4>
                    <div className="p-2 sm:p-3 bg-muted rounded-lg">
                      <p className="text-xs sm:text-sm line-clamp-3">{submission.prompt_text}</p>
                    </div>
                  </div>
                  {/* ...other fields... */}
                  <div className="flex gap-2 pt-2 flex-col sm:flex-row">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full sm:w-auto"
                      onClick={() => setViewingSubmission(submission)}
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      View Full
                    </Button>
                    {activeTab === 'PENDING' && (
                      <>
                        <Button
                          variant="default"
                          size="sm"
                          className="w-full sm:w-auto"
                          onClick={() => {
                            setReviewingSubmission(submission);
                            setReviewAction('approve');
                          }}
                        >
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Approve
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          className="w-full sm:w-auto"
                          onClick={() => {
                            setReviewingSubmission(submission);
                            setReviewAction('reject');
                          }}
                        >
                          <XCircle className="h-4 w-4 mr-1" />
                          Reject
                        </Button>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
      {/* Dialog for viewing submission */}
      <Dialog open={!!viewingSubmission} onOpenChange={() => setViewingSubmission(null)}>
        <DialogContent className="max-w-full sm:max-w-2xl p-2 sm:p-4">
          <DialogHeader>
            <DialogTitle className="text-base sm:text-lg">Full Submission</DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-80 sm:h-96 w-full rounded-md border p-2 sm:p-4">
            {viewingSubmission && (
              <div className="space-y-2 sm:space-y-4">
                <div>
                  <h4 className="font-medium mb-1 sm:mb-2 text-sm sm:text-base">Prompt Text:</h4>
                  <p className="text-xs sm:text-sm whitespace-pre-wrap">{viewingSubmission.prompt_text}</p>
                </div>
                {viewingSubmission.submission_notes && (
                  <div>
                    <h4 className="font-medium mb-1 sm:mb-2 text-sm sm:text-base">Submitter Notes:</h4>
                    <p className="text-xs sm:text-sm">{viewingSubmission.submission_notes}</p>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>
      {/* Dialog for approve/reject */}
      <Dialog open={!!reviewingSubmission} onOpenChange={() => setReviewingSubmission(null)}>
        <DialogContent className="max-w-full sm:max-w-md p-2 sm:p-4">
          <DialogHeader>
            <DialogTitle className="text-base sm:text-lg">
              {reviewAction === 'approve' ? 'Approve Submission' : 'Reject Submission'}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <Textarea
              placeholder={reviewAction === 'approve' ? 'Add approval notes (optional)' : 'Reason for rejection (optional)'}
              value={adminNotes}
              onChange={(e) => setAdminNotes(e.target.value)}
              className="text-xs sm:text-sm"
            />
            <div className="flex gap-2">
              <Button
                onClick={handleReview}
                disabled={isSubmitting}
                variant={reviewAction === 'approve' ? 'default' : 'destructive'}
              >
                {reviewAction === 'approve' ? 'Approve & Add to Library' : 'Reject Submission'}
              </Button>
              <Button
                variant="outline"
                onClick={() => setReviewingSubmission(null)}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}