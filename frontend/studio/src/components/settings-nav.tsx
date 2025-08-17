
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { User, Palette, KeyRound, Info, ArrowLeft } from 'lucide-react';

const navItems = [
  { href: '/settings', label: 'Profile', icon: User },
  { href: '/settings/appearance', label: 'Appearance', icon: Palette },
  { href: '/settings/account', label: 'Account', icon: KeyRound },
  { href: '/settings/about', label: 'About', icon: Info },
];

export function SettingsNav() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-col gap-1">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className={cn(
            'flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary',
            'dark:hover:text-sidebar-primary',
            pathname === item.href && 'bg-muted font-medium text-primary dark:bg-sidebar-accent dark:text-sidebar-primary'
          )}
        >
          <item.icon className="h-4 w-4" />
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
