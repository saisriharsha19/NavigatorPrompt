
'use client';

import * as React from 'react';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';

type User = {
  id: string;
  email: string;
  username: string;
  name: string;
  is_admin: boolean;
  is_student: boolean;
  is_faculty: boolean;
  is_staff: boolean;
};

type AuthContextType = {
  user: User | null;
  token: string | null;
  isAuthLoading: boolean;
  login: (email: string) => Promise<boolean>;
  logout: () => void;
};

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<User | null>(null);
  const [token, setToken] = React.useState<string | null>(null);
  const [isAuthLoading, setIsAuthLoading] = React.useState(true);
  const { toast } = useToast();
  const router = useRouter();

  const checkAuthStatus = React.useCallback(async () => {
    setIsAuthLoading(true);
    try {
        const storedToken = localStorage.getItem('auth_token');
        const storedUser = localStorage.getItem('auth_user');

        if (storedToken && storedUser) {
            setToken(storedToken);
            setUser(JSON.parse(storedUser));
        }
    } catch (error) {
        console.error("Failed to parse auth data from storage", error);
        // Clear corrupted data
        localStorage.removeItem('auth_user');
        localStorage.removeItem('auth_token');
        document.cookie = 'auth_token=; path=/; max-age=0;';
        document.cookie = 'auth_token_payload=; path=/; max-age=0;';
    } finally {
        setIsAuthLoading(false);
    }
  }, []);

  React.useEffect(() => {
    checkAuthStatus();
  }, [checkAuthStatus]);
  
  const login = async (email: string) => {
    setIsAuthLoading(true);
    try {
        if (!BACKEND_URL) {
            toast({
                variant: 'destructive',
                title: 'Configuration Error',
                description: 'The backend service URL is not configured.',
            });
            return false;
        }
        
        const response = await fetch(`${BACKEND_URL}/auth/mock/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to sign in.');
        }

        const data = await response.json();
        
        const userData: User = data.user;

        setUser(userData);
        setToken(data.access_token);
        localStorage.setItem('auth_user', JSON.stringify(userData));
        localStorage.setItem('auth_token', data.access_token);
        
        const cookieOptions = `path=/; max-age=86400; samesite=lax`;
        document.cookie = `auth_token=${data.access_token}; ${cookieOptions}`;
        document.cookie = `auth_token_payload=${encodeURIComponent(JSON.stringify(userData))}; ${cookieOptions}`;

        toast({
            title: 'Signed In',
            description: `Welcome, ${userData.name}!`,
        });

        router.push('/');
        router.refresh();
        return true;

    } catch (error: any) {
        toast({
            variant: 'destructive',
            title: 'Sign In Failed',
            description: error.message,
        });
        setUser(null);
        setToken(null);
        localStorage.removeItem('auth_user');
        localStorage.removeItem('auth_token');
        document.cookie = 'auth_token=; path=/; max-age=0;';
        document.cookie = 'auth_token_payload=; path=/; max-age=0;';
        return false;
    } finally {
        setIsAuthLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('auth_user');
    localStorage.removeItem('auth_token');
    document.cookie = 'auth_token=; path=/; max-age=0;';
    document.cookie = 'auth_token_payload=; path=/; max-age=0;';
    toast({
      title: 'Signed Out',
      description: 'You have been successfully signed out.',
    });
    router.push('/');
    router.refresh();
  };

  return (
    <AuthContext.Provider value={{ user, token, isAuthLoading, login, logout }}>
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
