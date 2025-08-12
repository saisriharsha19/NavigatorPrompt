
import { getAdminUsers } from "@/app/actions";
import { AdminUsersClient } from "@/components/admin-users-client";

export default async function AdminUsersPage() {
    const users = await getAdminUsers();

    return (
        <div>
            <h2 className="text-xl font-bold tracking-tight mb-4">User Management</h2>
            <AdminUsersClient initialUsers={users} />
        </div>
    )
}
