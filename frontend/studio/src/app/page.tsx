import { AppHeader } from '@/components/app-header';
import { PromptForgeClient } from '@/components/prompt-forge-client';
import { AppFooter } from '@/components/app-footer';

export default function Home() {
  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <AppHeader />
      <main className="flex-1">
        <div className="container mx-auto h-full max-w-7xl p-6 sm:p-8 md:p-10">
            <h1 className="sr-only">Navigator Sailor</h1>
            <PromptForgeClient />
        </div>
      </main>
      <AppFooter />
    </div>
  );
}
