
'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/hooks/use-auth';
import { Loader2 } from 'lucide-react';
import Cookies from 'js-cookie';
function AuthCallback() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { refreshSession } = useAuth();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('Processing authentication...');

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        // Check if there's an error in the URL params
        const error = searchParams.get('error');
        if (error) {
          setStatus('error');
          setMessage(`Authentication failed: ${error}`);
          setTimeout(() => router.push('/'), 3000);
          return;
        }

        // Check if there's a token in the URL params (from SAML callback)
        const token = searchParams.get('token');
        if (token) {
          Cookies.set('auth_token', token, { expires: 1 });
        }

        // Refresh the session to get user data
        await refreshSession();
        
        setStatus('success');
        setMessage('Authentication successful! Redirecting...');
        
        // Redirect to the intended page or home
        const redirectTo = searchParams.get('redirect') || '/';
        setTimeout(() => router.push(redirectTo), 1000);
        
      } catch (error: any) {
        console.error('Auth callback error:', error);
        setStatus('error');
        setMessage('Authentication failed. Please try again.');
        setTimeout(() => router.push('/'), 3000);
      }
    };

    handleAuthCallback();
  }, [searchParams, router, refreshSession]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="mx-auto max-w-md text-center">
        <div className="flex flex-col items-center space-y-4">
          {status === 'loading' && (
            <>
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <h1 className="text-xl font-semibold">Signing you in...</h1>
            </>
          )}
          
          {status === 'success' && (
            <>
              <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
                <svg className="h-5 w-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h1 className="text-xl font-semibold text-green-600">Success!</h1>
            </>
          )}
          
          {status === 'error' && (
            <>
              <div className="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center">
                <svg className="h-5 w-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <h1 className="text-xl font-semibold text-red-600">Authentication Failed</h1>
            </>
          )}
          
          <p className="text-muted-foreground">{message}</p>
        </div>
      </div>
    </div>
  );
}


export default function AuthCallbackPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AuthCallback />
    </Suspense>
  );
}
