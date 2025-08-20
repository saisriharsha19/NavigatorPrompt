'use client';

import { useToast } from '@/hooks/use-toast';
import * as React from 'react';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';

type User = {
  id: string;
  email: string;
  username: string;
  name?: string;
  is_admin: boolean;
  is_student: boolean;
  is_faculty: boolean;
  is_staff: boolean;
};

type AuthState = {
  isAuthenticated: boolean;
  user: User | null;
  isLoading: boolean;
  isAuthLoading: boolean;
  error: string | null;
};

type AuthContextType = AuthState & {
  userId: string | null;
  isAdmin: boolean;
  login: (email?: string) => Promise<boolean>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  showLoginModal: boolean;
  setShowLoginModal: (show: boolean) => void;
};

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Fix 1: Start with consistent initial state
  const [authState, setAuthState] = React.useState<AuthState>({
    isAuthenticated: false,
    user: null,
    isLoading: false, // Changed: Start with false to match server
    isAuthLoading: false,
    error: null,
  });
  
  // Fix 2: Track hydration state
  const [isHydrated, setIsHydrated] = React.useState(false);
  const { toast } = useToast();
  const router = useRouter();
  const [showLoginModal, setShowLoginModal] = React.useState(false);

  // Fix 3: Handle hydration
  React.useEffect(() => {
    setIsHydrated(true);
    // Only start loading after hydration
    setAuthState(prev => ({ ...prev, isLoading: true }));
  }, []);

  const checkAuthStatus = React.useCallback(async () => {
    // Only run after hydration
    if (!isHydrated) return;
    
    try {
      const token = Cookies.get('auth_token');
      if (!token) {
        setAuthState(prev => ({ ...prev, isAuthenticated: false, user: null, isLoading: false }));
        return;
      }

      const response = await fetch(`${BACKEND_URL}/auth/session`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const sessionData = await response.json();
        if (sessionData.isAuthenticated && sessionData.user) {
          setAuthState({
            isAuthenticated: true,
            user: sessionData.user,
            isLoading: false,
            isAuthLoading: false,
            error: null,
          });
          return;
        }
      }

      // Clear invalid token
      Cookies.remove('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        isAuthLoading: false,
        error: null,
      });
    } catch (error) {
      console.error('Auth check failed:', error);
      Cookies.remove('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        isAuthLoading: false,
        error: 'Authentication check failed',
      });
    }
  }, [isHydrated]);

  // Fix 4: Only check auth after hydration
  React.useEffect(() => {
    if (isHydrated) {
      checkAuthStatus();
    }
  }, [isHydrated, checkAuthStatus]);
  
  const login = React.useCallback(async (email?: string) => {
    setAuthState(prev => ({ ...prev, isAuthLoading: true, error: null }));

    try {
      // For production builds or when SAML is explicitly enabled, redirect to SAML login.
      if (process.env.NEXT_PUBLIC_ENABLE_DEV_LOGIN !== 'true') {
        // Ensure we're on client side before redirecting
        if (typeof window !== 'undefined') {
          window.location.href = `${BACKEND_URL}/auth/saml/login`;
        }
        return true;
      }

      // For development/testing, use the mock login flow.
      if (process.env.NEXT_PUBLIC_ENABLE_DEV_LOGIN === 'true') {
        // If no email is provided, show the modal to let them choose a mock user.
        if (typeof email !== 'string') {
          setShowLoginModal(true);
          setAuthState(prev => ({ ...prev, isAuthLoading: false }));
          return false;
        }

        // Email provided, proceed with mock login API call.
        const response = await fetch(`${BACKEND_URL}/auth/mock/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Login failed');
        }

        const data = await response.json();
        
        if (data.success && data.access_token) {
          // Set cookie with proper options for production
          Cookies.set('auth_token', data.access_token, { 
            expires: 7,
            secure: false,
            sameSite: 'lax',
            path: '/',
            domain: undefined
          });
          
          setAuthState({
            isAuthenticated: true,
            user: data.user,
            isLoading: false,
            isAuthLoading: false,
            error: null,
          });

          toast({
            title: 'Signed In Successfully',
            description: `Welcome back, ${data.user.name || data.user.username}!`,
          });
          
          // Close modal and refresh
          setShowLoginModal(false);
          router.refresh();
          return true;
        } else {
          throw new Error('Invalid response from server');
        }
      }

      // Fallback for production if dev login isn't enabled
      if (typeof window !== 'undefined') {
        window.location.href = `${BACKEND_URL}/auth/saml/login`;
      }
      return true;

    } catch (error: any) {
      console.error('Login failed:', error);
      setAuthState(prev => ({
        ...prev,
        isAuthLoading: false,
        error: error.message || 'Login failed',
      }));

      toast({
        variant: 'destructive',
        title: 'Sign In Failed',
        description: error.message || 'Unable to sign in. Please try again.',
      });
      return false;
    }
  }, [toast, router]);

  const logout = React.useCallback(async () => {
    try {
      const token = Cookies.get('auth_token');
      
      if (token) {
        await fetch(`${BACKEND_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }).catch(err => console.warn('Backend logout failed, proceeding with client-side cleanup:', err));
      }

      Cookies.remove('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        isAuthLoading: false,
        error: null,
      });

      toast({
        title: 'Signed Out',
        description: 'You have been signed out successfully.',
      });
      
      setShowLoginModal(false);
      router.push('/');
    } catch (error: any) {
      console.error('Logout error:', error);
      Cookies.remove('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        isAuthLoading: false,
        error: null,
      });
    }
  }, [toast, router]);

  const refreshSession = React.useCallback(async () => {
    if (isHydrated) {
      await checkAuthStatus();
    }
  }, [checkAuthStatus, isHydrated]);

  const userId = authState.user?.id || null;
  const isAdmin = authState.user?.is_admin || false;

  const contextValue: AuthContextType = {
    ...authState,
    userId,
    isAdmin,
    login,
    logout,
    refreshSession,
    showLoginModal,
    setShowLoginModal,
  };

  // Fix 5: Show loading during hydration
  if (!isHydrated) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-background">
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = React.useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}