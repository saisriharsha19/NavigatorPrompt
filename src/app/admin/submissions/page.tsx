
import { getAdminLibrarySubmissions } from "@/app/actions";
import { AdminSubmissionsClient } from "@/components/admin-submissions-client";
import { cookies } from 'next/headers';

export default async function AdminSubmissionsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const cookieStore = cookies();
  const token = cookieStore.get('auth_token')?.value;

  const initialStatus = (searchParams.status as 'PENDING' | 'APPROVED' | 'REJECTED') || 'PENDING';
  
  if (!token) {
    // Handle case where user is not authenticated.
    return (
        <div>
            <h2 className="text-xl font-bold tracking-tight mb-4">Library Submissions</h2>
            <p className="text-muted-foreground">You must be logged in as an admin to view submissions.</p>
        </div>
    );
  }

  // Fetch all statuses concurrently to provide to the client for instant filtering
  const [pending, approved, rejected] = await Promise.all([
    getAdminLibrarySubmissions('PENDING', token),
    getAdminLibrarySubmissions('APPROVED', token),
    getAdminLibrarySubmissions('REJECTED', token),
  ]);

  return (
    <div>
      <h2 className="text-xl font-bold tracking-tight mb-4">Library Submissions</h2>
      <AdminSubmissionsClient
        initialStatus={initialStatus}
        pendingSubmissions={pending}
        approvedSubmissions={approved}
        rejectedSubmissions={rejected}
      />
    </div>
  );
}
