
'use client';

import Link from 'next/link';
import { Menu, Moon, Sun, UserCircle, LogIn, LogOut, Settings, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Image from 'next/image';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { useAuth } from '@/hooks/use-auth';
import { useTheme } from 'next-themes';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { DevLoginModal } from '@/components/dev-login-modal';
import { useState } from 'react';
import { AdminNav } from './admin-nav';
import { SettingsNav } from './settings-nav';

export function AppHeader() {
  const { isAuthenticated, isLoading, user, login, logout, showLoginModal, setShowLoginModal } = useAuth();
  const { setTheme, theme } = useTheme();
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const mainNavItems = [
    { href: '/', label: 'Generator' },
    { href: '/history', label: 'History' },
    { href: '/library', label: 'Library' },
  ];

  const handleMobileSignIn = () => {
    setMobileMenuOpen(false);
    login();
  };

  const handleMobileSignOut = () => {
    setMobileMenuOpen(false);
    logout();
  };

  const isSettingsPage = pathname.startsWith('/settings');
  const isAdminPage = pathname.startsWith('/admin');

  const getMobileNavContent = () => {
    let navContent;
    let title = 'Main Menu';
    
    const mainNav = (
        <div className="space-y-2">
            {mainNavItems.map((item) => (
            <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                'flex items-center rounded-lg px-3 py-2 text-base font-medium transition-colors hover:bg-accent hover:text-accent-foreground',
                pathname === item.href && 'bg-accent text-accent-foreground'
                )}
            >
                {item.label}
            </Link>
            ))}
        </div>
    );

    if (isAdminPage) {
      title = 'Admin Menu';
      navContent = (
        <>
          {mainNav}
          <hr className="my-4" />
          <h3 className="px-3 text-sm font-medium text-muted-foreground mb-2">Admin</h3>
          <AdminNav />
        </>
      );
    } else if (isSettingsPage) {
      title = 'Settings Menu';
      navContent = (
        <>
          {mainNav}
          <hr className="my-4" />
          <h3 className="px-3 text-sm font-medium text-muted-foreground mb-2">Settings</h3>
          <SettingsNav />
        </>
      );
    } else {
        navContent = mainNav;
    }
    return { title, navContent };
  };

  const {title: mobileMenuTitle, navContent: mobileNavContent} = getMobileNavContent();


  return (
    <TooltipProvider>
      <header className="sticky top-0 z-50 flex h-14 sm:h-16 w-full shrink-0 items-center gap-2 sm:gap-4 border-b bg-card px-3 sm:px-6">
        
        <div className="flex items-center gap-2 sm:gap-4">
          {/* Mobile: Show hamburger */}
          <div className="md:hidden">
              <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="shrink-0 h-8 w-8 sm:h-10 sm:w-10">
                    <Menu className="h-4 w-4 sm:h-5 sm:w-5" />
                    <span className="sr-only">Toggle navigation menu</span>
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-[280px] sm:w-[300px] flex flex-col">
                  <SheetHeader>
                    <SheetTitle>
                        <Link 
                            href="/" 
                            className="flex items-center gap-2 text-lg font-semibold"
                            onClick={() => setMobileMenuOpen(false)}
                        >
                            <Image src="/NavGAI-19.png" width={25} height={25} alt="Navigator Logo" />
                            <span className="font-bold">Navigator Prompt</span>
                        </Link>
                    </SheetTitle>
                    <SheetDescription className="sr-only">{mobileMenuTitle}</SheetDescription>
                  </SheetHeader>
                  
                  <div className="py-4 flex-1" onClick={() => setMobileMenuOpen(false)}>
                    {mobileNavContent}
                  </div>
                  {/* Footer section of the sheet */}

                </SheetContent>
              </Sheet>
          </div>

          {/* Desktop: Show full nav */}
          <nav className="hidden flex-col gap-6 font-medium md:flex md:flex-row md:items-center md:gap-5 lg:gap-6">
            <Link href="/" className="flex items-center gap-2 text-lg font-semibold md:text-base">
              <Image src="/NavGAI-19.png" width={25} height={25} alt="Navigator Logo" />
              <h1 className="text-lg lg:text-xl font-bold tracking-tight">Navigator Prompt</h1>
            </Link>
            {mainNavItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'relative rounded-full px-3 py-1.5 text-sm text-muted-foreground transition-colors duration-300 hover:bg-muted',
                  pathname === item.href
                    ? 'text-accent-foreground'
                    : 'hover:text-foreground'
                )}
              >
                <AnimatePresence>
                  {pathname === item.href && (
                    <motion.span
                      className="absolute inset-0 z-0 rounded-full bg-accent"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1, transition: { duration: 0.25, ease: 'easeOut' } }}
                      exit={{ opacity: 0, transition: { duration: 0.1, ease: 'easeIn' } }}
                    />
                  )}
                </AnimatePresence>
                <span className="relative z-10">{item.label}</span>
              </Link>
            ))}
          </nav>
        </div>

        <div className="ml-auto flex items-center gap-1 sm:gap-2">
          {/* Theme Toggle - for both mobile and desktop */}
          <DropdownMenu>
            <Tooltip>
              <TooltipTrigger asChild>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8 sm:h-10 sm:w-10">
                    <Sun className="h-[1.1rem] w-[1.1rem] sm:h-[1.2rem] sm:w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                    <Moon className="absolute h-[1.1rem] w-[1.1rem] sm:h-[1.2rem] sm:w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                    <span className="sr-only">Toggle theme</span>
                  </Button>
                </DropdownMenuTrigger>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle Theme</p>
              </TooltipContent>
            </Tooltip>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme('light')}>Light</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme('dark')}>Dark</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme('system')}>System</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>


          {/* Mobile Auth - Show beside theme toggle on mobile */}
          <div className="sm:hidden">
            {isLoading ? (
              <Button variant="ghost" size="icon" disabled className="h-8 w-8">
                <UserCircle className="h-5 w-5 animate-pulse" />
              </Button>
            ) : isAuthenticated ? (
              <DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
                        <UserCircle className="h-5 w-5" />
                        <span className="sr-only">Toggle user menu</span>
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>My Account</p>
                  </TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium leading-none truncate">{user?.name || user?.username}</p>
                      <p className="text-xs leading-none text-muted-foreground truncate">{user?.email}</p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href="/settings" className="cursor-pointer">
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </Link>
                  </DropdownMenuItem>
                  {user?.is_admin && (
                    <DropdownMenuItem asChild>
                      <Link href="/admin" className="cursor-pointer">
                        <Shield className="mr-2 h-4 w-4" />
                        Admin Panel
                      </Link>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => logout()} className="cursor-pointer text-destructive focus:text-destructive">
                    <LogOut className="mr-2 h-4 w-4" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button onClick={() => login()} variant="ghost" size="icon" className="h-8 w-8">
                    <LogIn className="h-4 w-4" />
                    <span className="sr-only">Sign In</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Sign in to your account</p>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          {/* Desktop Auth */}
          <div className="hidden sm:block">
            {isLoading ? (
              <Button variant="ghost" size="icon" disabled className="h-8 w-8 sm:h-10 sm:w-10">
                <UserCircle className="h-5 w-5 sm:h-6 sm:w-6 animate-pulse" />
              </Button>
            ) : isAuthenticated ? (
              <DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="rounded-full h-8 w-8 sm:h-10 sm:w-10">
                        <UserCircle className="h-5 w-5 sm:h-6 sm:w-6" />
                        <span className="sr-only">Toggle user menu</span>
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>My Account</p>
                  </TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium leading-none truncate">{user?.name || user?.username}</p>
                      <p className="text-xs leading-none text-muted-foreground truncate">{user?.email}</p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href="/settings" className="cursor-pointer">
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </Link>
                  </DropdownMenuItem>
                  {user?.is_admin && (
                    <DropdownMenuItem asChild>
                      <Link href="/admin" className="cursor-pointer">
                        <Shield className="mr-2 h-4 w-4" />
                        Admin Panel
                      </Link>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => logout()} className="cursor-pointer text-destructive focus:text-destructive">
                    <LogOut className="mr-2 h-4 w-4" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button onClick={() => login()} variant="ghost" size="sm" className="flex items-center gap-2">
                    <LogIn className="h-4 w-4" />
                    <span className="hidden lg:inline">Sign In</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Sign in to your account</p>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </header>
      
      {process.env.NEXT_PUBLIC_ENABLE_DEV_LOGIN === 'true' && (
          <DevLoginModal
              isOpen={showLoginModal}
              onClose={() => setShowLoginModal(false)}
              onLogin={login}
          />
      )}
    </TooltipProvider>
  );
}
