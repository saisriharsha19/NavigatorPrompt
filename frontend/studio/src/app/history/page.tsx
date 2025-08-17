
import { AppHeader } from "@/components/app-header";
import { AppFooter } from "@/components/app-footer";
import { PromptHistoryClient } from "@/components/prompt-history-client";
import { getHistoryPromptsFromDB } from "@/app/actions";
import { cookies } from "next/headers";

export default async function HistoryPage() {
  const cookieStore = await cookies();
  const token = cookieStore.get('auth_token')?.value;
  
  // Fetch initial prompts on the server. The client hook will manage updates.
  const initialPrompts = token ? await getHistoryPromptsFromDB("dummy-user-id", token) : [];

  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <AppHeader />
      <main className="flex-1">
        <PromptHistoryClient initialPrompts={initialPrompts} />
      </main>
      <AppFooter />
    </div>
  );
}
