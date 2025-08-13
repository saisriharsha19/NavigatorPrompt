
import { AppHeader } from "@/components/app-header";
import { AppFooter } from "@/components/app-footer";
import { LibraryClient } from "@/components/library-client";
import { getLibraryPromptsFromDB } from "@/app/actions";
import { cookies } from "next/headers";

export default async function LibraryPage() {
  const cookieStore = await cookies();
  // We can get the user ID from the token if it exists to check for starred prompts
  const tokenPayloadCookie = cookieStore.get('auth_token_payload');
  const userId = tokenPayloadCookie ? JSON.parse(decodeURIComponent(tokenPayloadCookie.value)).id : undefined;

  const initialPrompts = await getLibraryPromptsFromDB(userId);
  
  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <AppHeader />
      <main className="flex-1">
        <LibraryClient initialPrompts={initialPrompts} />
      </main>
      <AppFooter />
    </div>
  );
}
