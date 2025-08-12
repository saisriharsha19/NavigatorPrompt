
import { getAdminLibrarySubmissions } from "@/app/actions";
import { AdminSubmissionsClient } from "@/components/admin-submissions-client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default async function AdminSubmissionsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const status = (searchParams.status as 'PENDING' | 'APPROVED' | 'REJECTED') || 'PENDING';
  
  const submissions = await getAdminLibrarySubmissions(status);

  return (
    <div>
      <h2 className="text-xl font-bold tracking-tight mb-4">Library Submissions</h2>
      <AdminSubmissionsClient initialSubmissions={submissions} initialStatus={status} />
    </div>
  );
}
