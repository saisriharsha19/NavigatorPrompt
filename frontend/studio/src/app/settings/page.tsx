
'use client';

import { useAuth } from '@/hooks/use-auth';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export default function ProfileSettingsPage() {
  const { user } = useAuth();

  if (!user) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your Profile</CardTitle>
        <CardDescription>
          This information is based on your university credentials.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" value={user.name || ''} disabled />
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" value={user.email || ''} disabled />
        </div>
        <div className="space-y-2">
          <Label htmlFor="userId">User ID</Label>
          <Input id="userId" value={user.id || 'N/A'} disabled />
        </div>
         <div className="space-y-2">
          <Label htmlFor="role">Affiliation</Label>
          <Input id="role" value={
            user.is_admin ? 'Admin' : 
            user.is_faculty ? 'Faculty' :
            user.is_student ? 'Student' :
            user.is_staff ? 'Staff' :
            'User'
          } disabled />
        </div>
      </CardContent>
    </Card>
  );
}
