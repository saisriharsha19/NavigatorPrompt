
'use client';

import Link from 'next/link';
import { Menu, Moon, Sun, UserCircle } from 'lucide-react';
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

export function AppHeader() {
  const { isAuthenticated, login, logout } = useAuth();
  const { setTheme } = useTheme();
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Generator' },
    { href: '/history', label: 'History' },
    { href: '/library', label: 'Library' },
  ];

  return (
    <TooltipProvider>
      <header className="sticky top-0 z-50 flex h-16 w-full shrink-0 items-center gap-4 border-b bg-card px-4 sm:px-6">
        
        <div className="flex items-center gap-4">
          {/* Mobile: Show nav */}
          <div className="md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="shrink-0">
                    <Menu className="h-5 w-5" />
                    <span className="sr-only">Toggle navigation menu</span>
                  </Button>
                </SheetTrigger>
                <SheetContent side="left">
                  <SheetHeader className="sr-only">
                    <SheetTitle>Navigation Menu</SheetTitle>
                    <SheetDescription>Select a page to navigate to.</SheetDescription>
                  </SheetHeader>
                  <nav className="grid gap-6 text-lg font-medium">
                    <Link href="/" className="flex items-center gap-2 text-lg font-semibold">
                      <Image src="/NavGAI-19.png" width={25} height={25} alt="Navigator Logo" />
                      <span className="font-bold">Navigator Prompt</span>
                    </Link>
                    {navItems.map((item) => (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={cn(
                          'text-muted-foreground hover:text-foreground',
                          pathname === item.href && 'font-semibold text-foreground'
                        )}
                      >
                        {item.label}
                      </Link>
                    ))}
                  </nav>
                </SheetContent>
              </Sheet>
          </div>

          {/* Desktop: Show full nav */}
          <nav className="hidden flex-col gap-6 font-medium md:flex md:flex-row md:items-center md:gap-5 lg:gap-6">
            <Link href="/" className="flex items-center gap-2 text-lg font-semibold md:text-base">
              <Image src="/NavGAI-19.png" width={25} height={25} alt="Navigator Logo" />
              <h1 className="text-xl font-bold tracking-tight">Navigator Prompt</h1>
            </Link>
            {navItems.map((item) => (
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

        <div className="ml-auto flex items-center gap-2">
          <DropdownMenu>
            <Tooltip>
              <TooltipTrigger asChild>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                    <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
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

          <DropdownMenu>
            <Tooltip>
              <TooltipTrigger asChild>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="rounded-full">
                    <UserCircle className="h-6 w-6" />
                    <span className="sr-only">Toggle user menu</span>
                  </Button>
                </DropdownMenuTrigger>
              </TooltipTrigger>
              <TooltipContent>
                <p>My Account</p>
              </TooltipContent>
            </Tooltip>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {isAuthenticated ? (
                <>
                  <DropdownMenuItem asChild>
                    <Link href="/settings" className="cursor-pointer">Settings</Link>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => logout()} className="cursor-pointer">Sign Out</DropdownMenuItem>
                </>
              ) : (
                <DropdownMenuItem onClick={() => login()} className="cursor-pointer">Sign In</DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>
    </TooltipProvider>
  );
}
