// Your complete app/layout.tsx should look like this:

import type {Metadata, Viewport} from 'next';
import './globals.css';
import { Toaster } from "@/components/ui/toaster"
import { AuthProvider } from '@/hooks/use-auth';
import { ThemeProvider } from '@/components/theme-provider';
import { PromptHistoryProvider } from '@/hooks/use-prompts';
import { PromptForgeProvider } from '@/hooks/use-prompt-forge';
import { LibraryProvider } from '@/hooks/use-library';
import { RateLimitIndicator } from '@/components/rate-limit-indicator';

export const metadata: Metadata = {
  title: 'Navigator Prompt',
  description: 'Create and iterate on prompts.',
  icons: {
    icon: [
      { url: '/favicon.ico?v=3' },
      { url: '/NavGAI-19.png?v=3', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/NavGAI-19.png?v=3', sizes: '180x180', type: 'image/png' },
    ],
  },
  manifest: '/manifest.json?v=3',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'Navigator Prompt',
  },
};

// Add the viewport export to fix the warning
export const viewport: Viewport = {
  themeColor: '#f97316',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"></link>
      </head>
      <body className="font-body antialiased flex flex-col min-h-screen">
        <AuthProvider>
          <PromptHistoryProvider>
            <LibraryProvider>
              <PromptForgeProvider>
                <ThemeProvider
                  attribute="class"
                  defaultTheme="system"
                  enableSystem
                  disableTransitionOnChange
                >
                  {children}
                  <RateLimitIndicator />
                </ThemeProvider>
              </PromptForgeProvider>
            </LibraryProvider>
          </PromptHistoryProvider>
        </AuthProvider>
        <Toaster />
      </body>
    </html>
  );
}