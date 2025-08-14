// frontend/studio/src/components/admin-submissions-client.tsx
'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { CheckCircle, XCircle, Clock, Eye, MessageSquare } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/hooks/use-auth';
import { getAdminLibrarySubmissions, reviewLibrarySubmission, LibrarySubmission } from '@/app/actions';
import { ScrollArea } from '@/components/ui/scroll-area';
import { formatDistanceToNow } from 'date-fns';

export function AdminSubmissionsClient() {
  const [allSubmissions, setAllSubmissions] = useState<LibrarySubmission[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'PENDING' | 'APPROVED' | 'REJECTED'>('PENDING');
  const [reviewingId, setReviewingId] = useState<string | null>(null);
  const [adminNotes, setAdminNotes] = useState('');
  const [viewingSubmission, setViewingSubmission] = useState<LibrarySubmission | null>(null);
  const { toast } = useToast();
  const { isAuthenticated } = useAuth();

  const loadAllSubmissions = async () => {
    if (!isAuthenticated) return;
    
    try {
      setIsLoading(true);
      const token = localStorage.getItem('auth_token');
      // Load all submissions without filter to get counts
      const data = await getAdminLibrarySubmissions(token);
      setAllSubmissions(data);
    } catch (error) {
      console.error('Failed to load submissions:', error);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Failed to load submissions.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAllSubmissions();
  }, [isAuthenticated]);

  const handleTabChange = (value: string) => {
    setActiveTab(value as 'PENDING' | 'APPROVED' | 'REJECTED');
  };

  const handleReview = async (submissionId: string, action: 'approve' | 'reject') => {
    try {
      setReviewingId(submissionId);
      const token = localStorage.getItem('auth_token');
      await reviewLibrarySubmission(submissionId, action, adminNotes, token);
      
      toast({
        title: 'Success',
        description: `Submission ${action}d successfully.`,
      });
      
      setAdminNotes('');
      await loadAllSubmissions(); // Reload all submissions
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message || `Failed to ${action} submission.`,
      });
    } finally {
      setReviewingId(null);
    }
  };

  const pendingCount = allSubmissions.filter(s => s.status === 'PENDING').length;
  const approvedCount = allSubmissions.filter(s => s.status === 'APPROVED').length;
  const rejectedCount = allSubmissions.filter(s => s.status === 'REJECTED').length;

  // Filter submissions for active tab
  const currentTabSubmissions = allSubmissions.filter(sub => sub.status === activeTab);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'APPROVED':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'REJECTED':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'PENDING':
        return <Clock className="h-4 w-4 text-yellow-600" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'APPROVED':
        return <Badge variant="default">Approved</Badge>;
      case 'REJECTED':
        return <Badge variant="destructive">Rejected</Badge>;
      case 'PENDING':
        return <Badge variant="secondary">Pending</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Library Submissions</h2>
        <div className="grid gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-4 bg-muted rounded w-3/4"></div>
                <div className="h-3 bg-muted rounded w-1/2"></div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="h-3 bg-muted rounded"></div>
                  <div className="h-3 bg-muted rounded w-5/6"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Library Submissions</h2>
        <Button variant="outline" onClick={loadAllSubmissions}>
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="PENDING">
            Pending ({pendingCount})
          </TabsTrigger>
          <TabsTrigger value="APPROVED">
            Approved ({approvedCount})
          </TabsTrigger>
          <TabsTrigger value="REJECTED">
            Rejected ({rejectedCount})
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="space-y-4">
          {currentTabSubmissions.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <p className="text-muted-foreground">No {activeTab.toLowerCase()} submissions found.</p>
              </CardContent>
            </Card>
          ) : (
            currentTabSubmissions.map((submission) => (
              <Card key={submission.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="flex items-center gap-2">
                        {getStatusIcon(submission.status)}
                        Submission from {submission.user_email || 'Unknown User'}
                      </CardTitle>
                      <CardDescription>
                        Submitted {formatDistanceToNow(new Date(submission.created_at), { addSuffix: true })}
                        {submission.reviewed_at && (
                          <> • Reviewed {formatDistanceToNow(new Date(submission.reviewed_at), { addSuffix: true })}</>
                        )}
                      </CardDescription>
                    </div>
                    {getStatusBadge(submission.status)}
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Prompt Text:</h4>
                    <div className="p-3 bg-muted rounded-lg">
                      <p className="text-sm line-clamp-3">{submission.prompt_text}</p>
                    </div>
                  </div>

                  {submission.submission_notes && (
                    <div>
                      <h4 className="font-medium mb-2">Submitter Notes:</h4>
                      <p className="text-sm text-muted-foreground">{submission.submission_notes}</p>
                    </div>
                  )}

                  {submission.admin_notes && (
                    <div>
                      <h4 className="font-medium mb-2">Admin Notes:</h4>
                      <p className="text-sm text-muted-foreground">{submission.admin_notes}</p>
                    </div>
                  )}

                  {submission.tags && submission.tags.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">Tags:</h4>
                      <div className="flex flex-wrap gap-1">
                        {submission.tags.map((tag, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 pt-2">
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm" onClick={() => setViewingSubmission(submission)}>
                          <Eye className="h-4 w-4 mr-1" />
                          View Full
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="sm:max-w-2xl">
                        <DialogHeader>
                          <DialogTitle>Full Submission</DialogTitle>
                        </DialogHeader>
                        <ScrollArea className="h-96 w-full rounded-md border p-4">
                          <div className="space-y-4">
                            <div>
                              <h4 className="font-medium mb-2">Prompt Text:</h4>
                              <p className="text-sm whitespace-pre-wrap">{submission.prompt_text}</p>
                            </div>
                            {submission.submission_notes && (
                              <div>
                                <h4 className="font-medium mb-2">Submitter Notes:</h4>
                                <p className="text-sm">{submission.submission_notes}</p>
                              </div>
                            )}
                          </div>
                        </ScrollArea>
                      </DialogContent>
                    </Dialog>

                    {submission.status === 'PENDING' && (
                      <>
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="default" size="sm" disabled={reviewingId === submission.id}>
                              <CheckCircle className="h-4 w-4 mr-1" />
                              Approve
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Approve Submission</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                              <Textarea
                                placeholder="Add approval notes (optional)"
                                value={adminNotes}
                                onChange={(e) => setAdminNotes(e.target.value)}
                              />
                              <div className="flex gap-2">
                                <Button 
                                  onClick={() => handleReview(submission.id, 'approve')}
                                  disabled={reviewingId === submission.id}
                                >
                                  Approve & Add to Library
                                </Button>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>

                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="destructive" size="sm" disabled={reviewingId === submission.id}>
                              <XCircle className="h-4 w-4 mr-1" />
                              Reject
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Reject Submission</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                              <Textarea
                                placeholder="Reason for rejection (optional)"
                                value={adminNotes}
                                onChange={(e) => setAdminNotes(e.target.value)}
                              />
                              <div className="flex gap-2">
                                <Button 
                                  variant="destructive"
                                  onClick={() => handleReview(submission.id, 'reject')}
                                  disabled={reviewingId === submission.id}
                                >
                                  Reject Submission
                                </Button>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}