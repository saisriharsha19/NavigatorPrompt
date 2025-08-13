
'use client';

import { useAuth } from '@/hooks/use-auth';
import { AppHeader } from '@/components/app-header';
import { AppFooter } from '@/components/app-footer';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from '@/components/ui/sheet';
import { SettingsNav } from '@/components/settings-nav';
import { Button } from '@/components/ui/button';
import { Menu, UserCircle } from 'lucide-react';
import * as React from 'react';
import {
  Dialog,
  DialogTrigger,
} from '@/components/ui/dialog';
import { MockLoginDialog } from '@/components/mock-login-dialog';


export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user } = useAuth();
  const [isLoginOpen, setIsLoginOpen] = React.useState(false);

  return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <AppHeader />
        <div className="flex flex-1 overflow-hidden">
          <aside className="hidden md:flex md:w-64 md:flex-col md:border-r">
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
             {/* Mobile-only header with trigger */}
            <header className="sticky top-16 z-10 flex h-14 items-center gap-4 border-b bg-background px-4 md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Menu className="h-5 w-5" />
                    <span className="sr-only">Toggle Settings Menu</span>
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-full max-w-xs">
                    <div className="flex flex-col gap-2 p-4 pt-8">
                        <h2 className="text-lg font-semibold tracking-tight">Settings</h2>
                    </div>
                    <nav className="grid gap-2 p-4 text-sm font-medium">
                      <SettingsNav />
                    </nav>
                </SheetContent>
              </Sheet>
              <h1 className="text-lg font-semibold">Settings</h1>
            </header>

            {user ? (
              <div className="p-6 sm:p-8 md:p-10">{children}</div>
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-center">
                <div className="flex max-w-md flex-col items-center rounded-lg border-2 border-dashed bg-muted/50 p-6 sm:p-8 lg:p-12">
                  <UserCircle className="h-12 w-12 text-muted-foreground sm:h-16 sm:w-16" />
                  <h2 className="mt-6 text-xl font-semibold tracking-tight sm:text-2xl">Access Your Settings</h2>
                  <p className="mt-2 text-muted-foreground">
                    Sign in with your GatorLink to manage your profile, appearance, and account settings.
                  </p>
                   <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
                        <DialogTrigger asChild>
                           <Button className="mt-6">Sign In with GatorLink</Button>
                        </DialogTrigger>
                        <MockLoginDialog onSuccess={() => setIsLoginOpen(false)} />
                    </Dialog>
                </div>
              </div>
            )}
          </main>
        </div>
        <AppFooter />
      </div>
  );
}
