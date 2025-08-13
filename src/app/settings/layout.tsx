
'use client';

import { useAuth } from '@/hooks/use-auth';
import { AppHeader } from '@/components/app-header';
import { AppFooter } from '@/components/app-footer';
import { Button } from '@/components/ui/button';
import { UserCircle } from 'lucide-react';
import { SettingsNav } from '@/components/settings-nav';
import * as React from 'react';

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, login } = useAuth();

  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <AppHeader />
      <main className="flex-1 bg-muted/40 py-8 md:py-10">
        <div className="container max-w-6xl px-4 sm:px-6 lg:px-8">
            <div className="mb-8 space-y-0.5">
                <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Settings</h1>
                <p className="text-muted-foreground">
                    Manage your account settings and preferences.
                </p>
            </div>
            {user ? (
              <div className="grid grid-cols-1 gap-8 md:grid-cols-[240px_1fr] md:gap-10">
                <aside className="md:mb-0">
                    <SettingsNav />
                </aside>
                <div className="grid gap-6">
                    {children}
                </div>
              </div>
            ) : (
            <div className="mx-auto max-w-lg flex flex-col h-full min-h-[40vh] items-center justify-center rounded-lg border-2 border-dashed bg-card p-6 text-center sm:p-8 lg:p-12">
                <UserCircle className="h-12 w-12 text-muted-foreground sm:h-16 sm:w-16" />
                <h2 className="mt-6 text-xl font-semibold tracking-tight sm:text-2xl">Access Your Settings</h2>
                <p className="mt-2 text-muted-foreground">
                Sign in to manage your profile, appearance, and account settings.
                </p>
                <p className="mt-4 text-sm text-muted-foreground">
                    Use the Sign In button in the header.
                </p>
            </div>
            )}
        </div>
      </main>
      <AppFooter />
    </div>
  );
}
