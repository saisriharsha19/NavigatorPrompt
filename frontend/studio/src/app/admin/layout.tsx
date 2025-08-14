// frontend/studio/src/app/admin/layout.tsx
'use client';

import { useAuth } from '@/hooks/use-auth';
import { AppHeader } from '@/components/app-header';
import { AppFooter } from '@/components/app-footer';
import { Button } from '@/components/ui/button';
import { ShieldAlert, UserCircle } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import * as React from 'react';

const adminNavItems = [
  { href: '/admin', label: 'Dashboard' },
  { href: '/admin/users', label: 'Users' },
  { href: '/admin/submissions', label: 'Submissions' },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isAuthenticated, isAdmin, login, isLoading } = useAuth();
  const pathname = usePathname();

  if (isLoading) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <AppHeader />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
            <p className="mt-2 text-muted-foreground">Loading...</p>
          </div>
        </main>
        <AppFooter />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <AppHeader />
        <main className="flex-1 bg-muted/40 py-8 md:py-10">
          <div className="container max-w-6xl px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-lg flex flex-col h-full min-h-[40vh] items-center justify-center rounded-lg border-2 border-dashed bg-card p-6 text-center sm:p-8 lg:p-12">
              <UserCircle className="h-12 w-12 text-muted-foreground sm:h-16 sm:w-16" />
              <h2 className="mt-6 text-xl font-semibold tracking-tight sm:text-2xl">Admin Access Required</h2>
              <p className="mt-2 text-muted-foreground">
                Sign in with admin credentials to access the admin panel.
              </p>
              <div className="mt-6 space-y-2">
                <Button onClick={login} className="w-full">
                  Sign In
                </Button>
                <p className="text-xs text-muted-foreground">
                  For testing, use: <code className="bg-muted px-1 rounded">admin@ufl.edu</code>
                </p>
              </div>
            </div>
          </div>
        </main>
        <AppFooter />
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <AppHeader />
        <main className="flex-1 bg-muted/40 py-8 md:py-10">
          <div className="container max-w-6xl px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-lg flex flex-col h-full min-h-[40vh] items-center justify-center rounded-lg border-2 border-dashed bg-card p-6 text-center sm:p-8 lg:p-12">
              <ShieldAlert className="h-12 w-12 text-destructive sm:h-16 sm:w-16" />
              <h2 className="mt-6 text-xl font-semibold tracking-tight sm:text-2xl">Access Denied</h2>
              <p className="mt-2 text-muted-foreground">
                You don't have admin privileges to access this area.
              </p>
              <Button asChild className="mt-6">
                <Link href="/">Return to Home</Link>
              </Button>
            </div>
          </div>
        </main>
        <AppFooter />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <AppHeader />
      <main className="flex-1 bg-muted/40 py-8 md:py-10">
        <div className="container max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="mb-8 space-y-0.5">
            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Admin Panel</h1>
            <p className="text-muted-foreground">
              Manage users, review submissions, and monitor platform activity.
            </p>
          </div>
          
          <div className="grid grid-cols-1 gap-8 md:grid-cols-[240px_1fr] md:gap-10">
            <aside className="md:mb-0">
              <nav className="flex flex-col gap-1">
                {adminNavItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      'flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary',
                      pathname === item.href && 'bg-muted font-medium text-primary'
                    )}
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
            </aside>
            
            <div className="grid gap-6">
              {children}
            </div>
          </div>
        </div>
      </main>
      <AppFooter />
    </div>
  );
}