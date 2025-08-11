
'use client';

import { useToast } from '@/hooks/use-toast';
import * as React from 'react';

type AuthContextType = {
  isAuthenticated: boolean;
  userId: string | null;
  isAdmin: boolean;
  login: () => void;
  logout: () => void;
};

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = React.useState(true); // Default to logged in
  const { toast } = useToast();

  // For this mock, we'll use a static user ID and admin status.
  const userId = 'mock-user-123';
  const isAdmin = true; // Set to true for full access during development

  const login = () => {
    setIsAuthenticated(true);
    toast({
      title: 'Logged In (Mock)',
      description: 'You are now signed in with a mock user account.',
    });
  };

  const logout = () => {
    setIsAuthenticated(false);
    toast({
      title: 'Logged Out (Mock)',
      description: 'You have been signed out.',
    });
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, userId, isAdmin, login, logout }}>
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
