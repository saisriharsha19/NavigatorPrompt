
import { getAdminUsers } from "@/app/actions";
import { AdminUsersClient } from "@/components/admin-users-client";
import { cookies } from 'next/headers';


export default async function AdminUsersPage() {
    const cookieStore = cookies();
    const token = cookieStore.get('auth_token')?.value;

    if (!token) {
      return (
          <div>
              <h2 className="text-xl font-bold tracking-tight mb-4">User Management</h2>
              <p className="text-muted-foreground">You must be logged in as an admin to view users.</p>
          </div>
      );
    }
    const users = await getAdminUsers(token);

    return (
        <div>
            <h2 className="text-xl font-bold tracking-tight mb-4">User Management</h2>
            <AdminUsersClient initialUsers={users} />
        </div>
    )
}
