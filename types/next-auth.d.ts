// This file is no longer needed for mock authentication, but can be kept for future next-auth integration.
import 'next-auth';

declare module 'next-auth' {
  interface Session {
    user?: {
      id: string;
    } & DefaultSession['user'];
  }
}
