// frontend/studio/src/components/dev-login-modal.tsx
'use client';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { UserCircle, Shield, GraduationCap, BookOpen } from 'lucide-react';

type MockUser = {
  email: string;
  name: string;
  role: string;
  icon: React.ReactNode;
  description: string;
  badges: string[];
};

const mockUsers: MockUser[] = [
  {
    email: 'admin@ufl.edu',
    name: 'Admin User',
    role: 'Administrator',
    icon: <Shield className="h-5 w-5" />,
    description: 'Full admin access to all features including /admin panel',
    badges: ['Admin', 'Staff']
  },
  {
    email: 'faculty@ufl.edu',
    name: 'Dr. Jane Professor',
    role: 'Faculty Member',
    icon: <GraduationCap className="h-5 w-5" />,
    description: 'Faculty member with teaching access',
    badges: ['Faculty']
  },
  {
    email: 'student@ufl.edu',
    name: 'John Student',
    role: 'Student',
    icon: <BookOpen className="h-5 w-5" />,
    description: 'Student with basic access',
    badges: ['Student']
  },
  {
    email: 'demo@example.com',
    name: 'Demo User',
    role: 'Demo Account',
    icon: <UserCircle className="h-5 w-5" />,
    description: 'Demo account for testing',
    badges: ['Demo']
  }
];

interface DevLoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLogin: (email: string) => void;
}

export function DevLoginModal({ isOpen, onClose, onLogin }: DevLoginModalProps) {
  const handleLogin = (email: string) => {
    onLogin(email);
    onClose();
  };

  if (process.env.NODE_ENV === 'production') {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Development Login</DialogTitle>
          <DialogDescription>
            Choose a user type to login as for testing purposes.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          {mockUsers.map((user) => (
            <button
              key={user.email}
              onClick={() => handleLogin(user.email)}
              className="w-full p-4 text-left rounded-lg border hover:bg-muted transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 text-muted-foreground">
                  {user.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium text-sm">{user.name}</h4>
                    <div className="flex gap-1">
                      {user.badges.map((badge) => (
                        <Badge key={badge} variant="secondary" className="text-xs">
                          {badge}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mb-1">{user.email}</p>
                  <p className="text-xs text-muted-foreground">{user.description}</p>
                </div>
              </div>
            </button>
          ))}
        </div>
        <div className="mt-4 p-3 bg-muted rounded-lg">
          <p className="text-xs text-muted-foreground">
            💡 <strong>Tip:</strong> Use <code>admin@ufl.edu</code> to access the admin panel at <code>/admin</code>
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}