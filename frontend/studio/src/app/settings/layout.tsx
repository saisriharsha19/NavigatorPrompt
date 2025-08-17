
'use client';

import { useAuth } from '@/hooks/use-auth';
import { AppHeader } from '@/components/app-header';
import { AppFooter } from '@/components/app-footer';
import { SettingsNav } from '@/components/settings-nav';
import { Button } from '@/components/ui/button';
import { UserCircle } from 'lucide-react';
import * as React from 'react';
import { DevLoginModal } from '@/components/dev-login-modal';


export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, login, showLoginModal, setShowLoginModal } = useAuth();

  return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <AppHeader />
        <div className="flex flex-1 overflow-hidden">
          <aside className="hidden w-64 flex-col border-r bg-background md:flex">
            <div className="flex flex-col gap-2 p-4 pt-8">
              <h2 className="text-lg font-semibold tracking-tight">Settings</h2>
            </div>
            <div className="flex-1 overflow-auto py-2">
              <nav className="grid items-start px-4 text-sm font-medium">
                <SettingsNav />
              </nav>
            </div>
          </aside>
          
          <main className="flex-1 overflow-y-auto">
            <div className="p-6 sm:p-8 md:p-10">
              {user ? (
                children
              ) : (
                <div className="flex h-full items-center justify-center p-6 text-center">
                  <div className="flex max-w-md flex-col items-center rounded-lg border-2 border-dashed bg-muted/50 p-6 sm:p-8 lg:p-12">
                    <UserCircle className="h-12 w-12 text-muted-foreground sm:h-16 sm:w-16" />
                    <h2 className="mt-6 text-xl font-semibold tracking-tight sm:text-2xl">Access Your Settings</h2>
                    <p className="mt-2 text-muted-foreground">
                      Sign in with your GatorLink to manage your profile, appearance, and account settings.
                    </p>
                    <Button onClick={() => login()} className="mt-6">Sign In with GatorLink</Button>
                  </div>
                </div>
              )}
            </div>
          </main>
        </div>
        <AppFooter />

        {process.env.NEXT_PUBLIC_ENABLE_DEV_LOGIN === 'true' && (
          <DevLoginModal
              isOpen={showLoginModal}
              onClose={() => setShowLoginModal(false)}
              onLogin={login}
          />
        )}
      </div>
  );
}
