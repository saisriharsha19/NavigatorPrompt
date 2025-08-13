
'use client';

import * as React from 'react';
import {
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from './ui/button';
import { useAuth } from '@/hooks/use-auth';

type MockLoginDialogProps = {
  onSuccess?: () => void;
};

export function MockLoginDialog({ onSuccess }: MockLoginDialogProps) {
  const { login, isAuthLoading } = useAuth();
  const [email, setEmail] = React.useState('student@ufl.edu');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const success = await login(email);
    if (success && onSuccess) {
      onSuccess();
    }
  };

  return (
    <DialogContent className="sm:max-w-[425px]">
      <form onSubmit={handleLogin}>
        <DialogHeader>
          <DialogTitle>Mock Sign In</DialogTitle>
          <DialogDescription>
            Enter an email to simulate logging in as different user types. Use `admin@ufl.edu` for admin access.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="email" className="text-right">
              Email
            </Label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="col-span-3"
              placeholder="e.g., student@ufl.edu"
            />
          </div>
        </div>
        <DialogFooter>
          <Button type="submit" disabled={isAuthLoading}>
            {isAuthLoading ? 'Signing In...' : 'Sign In'}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}
