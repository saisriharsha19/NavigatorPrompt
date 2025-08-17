
'use client';

import { useAuth } from '@/hooks/use-auth';
import { AppHeader } from '@/components/app-header';
import { AppFooter } from '@/components/app-footer';
import { Button } from '@/components/ui/button';
import { ShieldAlert, UserCircle } from 'lucide-react';
import Link from 'next/link';
import * as React from 'react';
import { AdminNav } from '@/components/admin-nav';

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isAuthenticated, isAdmin, login, isLoading } = useAuth();

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
                <Button onClick={() => login()} className="w-full">
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
      <div className="flex flex-1">
        <aside className="hidden w-64 flex-col border-r bg-background md:flex">
          <div className="flex flex-col gap-2 p-4 pt-8">
            <h2 className="text-lg font-semibold tracking-tight">Admin Panel</h2>
            <p className="text-sm text-muted-foreground">Manage the platform.</p>
          </div>
          <div className="flex-1 overflow-auto py-2">
            <nav className="grid items-start px-4 text-sm font-medium">
              <AdminNav />
            </nav>
          </div>
        </aside>

        <main className="flex-1 overflow-y-auto bg-muted/40">
          <div className="p-6 sm:p-8 md:p-10">
            {children}
          </div>
        </main>
      </div>
      <AppFooter />
    </div>
  );
}
