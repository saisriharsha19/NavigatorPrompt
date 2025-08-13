
import { getAdminLibrarySubmissions } from "@/app/actions";
import { AdminSubmissionsClient } from "@/components/admin-submissions-client";

export default async function AdminSubmissionsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const initialStatus = (searchParams.status as 'PENDING' | 'APPROVED' | 'REJECTED') || 'PENDING';
  
  // Fetch all statuses concurrently to provide to the client for instant filtering
  const [pending, approved, rejected] = await Promise.all([
    getAdminLibrarySubmissions('PENDING'),
    getAdminLibrarySubmissions('APPROVED'),
    getAdminLibrarySubmissions('REJECTED'),
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
