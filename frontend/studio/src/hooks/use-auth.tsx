'use client';

import { useToast } from '@/hooks/use-toast';
import * as React from 'react';

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
  error: string | null;
};

type AuthContextType = AuthState & {
  userId: string | null;
  isAdmin: boolean;
  login: (email?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  showLoginModal: boolean;
  setShowLoginModal: (show: boolean) => void;
};

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authState, setAuthState] = React.useState<AuthState>({
    isAuthenticated: false,
    user: null,
    isLoading: true,
    error: null,
  });
  const { toast } = useToast();

  // Check for existing token and validate session
  const checkExistingAuth = React.useCallback(async () => {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
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
            error: null,
          });
          return;
        }
      }

      // Invalid token, remove it
      localStorage.removeItem('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      console.error('Auth check failed:', error);
      localStorage.removeItem('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: 'Authentication check failed',
      });
    }
  }, []);

  // Initialize auth state on mount
  React.useEffect(() => {
    checkExistingAuth();
  }, [checkExistingAuth]);

  const [showLoginModal, setShowLoginModal] = React.useState(false);

  const login = React.useCallback(async (email?: string) => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true, error: null }));

      // Check if SAML is available (production)
      if (process.env.NODE_ENV === 'production' || process.env.NEXT_PUBLIC_USE_SAML === 'true') {
        // Redirect to SAML login
        window.location.href = `${BACKEND_URL}/auth/saml/login`;
        return;
      }

      // Development: Show modal if no email provided
      if (!email) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        setShowLoginModal(true);
        return;
      }

      // Development: Use mock login with provided email
      const response = await fetch(`${BACKEND_URL}/auth/mock/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email }), // Fix: ensure email is a string
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();
      
      if (data.success && data.access_token) {
        localStorage.setItem('auth_token', data.access_token);
        setAuthState({
          isAuthenticated: true,
          user: data.user,
          isLoading: false,
          error: null,
        });

        toast({
          title: 'Signed In Successfully',
          description: `Welcome back, ${data.user.name || data.user.username}!`,
        });
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error: any) {
      console.error('Login failed:', error);
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: error.message || 'Login failed',
      });

      toast({
        variant: 'destructive',
        title: 'Sign In Failed',
        description: error.message || 'Unable to sign in. Please try again.',
      });
    }
  }, [toast]);

  const logout = React.useCallback(async () => {
    try {
      const token = localStorage.getItem('auth_token');
      
      // Call backend logout endpoint if token exists
      if (token) {
        try {
          await fetch(`${BACKEND_URL}/auth/logout`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          });
        } catch (error) {
          // Ignore logout endpoint errors, proceed with local cleanup
          console.warn('Backend logout failed:', error);
        }
      }

      // Clear local storage and state
      localStorage.removeItem('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: null,
      });

      toast({
        title: 'Signed Out',
        description: 'You have been signed out successfully.',
      });
    } catch (error: any) {
      console.error('Logout error:', error);
      // Still clear local state even if logout call fails
      localStorage.removeItem('auth_token');
      setAuthState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: null,
      });
    }
  }, [toast]);

  const refreshSession = React.useCallback(async () => {
    await checkExistingAuth();
  }, [checkExistingAuth]);

  // Derive computed values
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