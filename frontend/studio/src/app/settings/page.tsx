'use client';

import { useAuth } from '@/hooks/use-auth';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export default function ProfileSettingsPage() {
  const { userId } = useAuth();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your Profile</CardTitle>
        <CardDescription>
          This is your mock user profile. In a real application, this would be based on your university credentials.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" defaultValue={'Mock User'} disabled />
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" defaultValue={'mock.user@example.com'} disabled />
        </div>
        <div className="space-y-2">
          <Label htmlFor="userId">User ID</Label>
          <Input id="userId" defaultValue={userId ?? 'N/A'} disabled />
        </div>
      </CardContent>
    </Card>
  );
}
