// frontend/studio/src/components/rate-limit-indicator.tsx - MODERN BOTTOM STYLE
'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle, X, Timer, Clock } from 'lucide-react';

interface RateLimitInfo {
  limit: number;
  remaining: number;
  resetTime: number;
  window: number;
  retryAfter?: number;
}

export function RateLimitIndicator() {
  const [rateLimitInfo, setRateLimitInfo] = useState<RateLimitInfo | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [countdown, setCountdown] = useState(0);

  // Countdown timer
  useEffect(() => {
    if (isVisible && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(prev => prev - 1);
      }, 1000);
      
      return () => clearTimeout(timer);
    } else if (isVisible && countdown <= 0) {
      setIsVisible(false);
    }
  }, [countdown, isVisible]);

  // Show rate limit UI
  const showRateLimitUI = (retryAfter: number = 60) => {
    const info: RateLimitInfo = {
      limit: 30,
      remaining: 0,
      resetTime: Math.floor(Date.now() / 1000) + retryAfter,
      window: 60,
      retryAfter: retryAfter
    };
    
    setRateLimitInfo(info);
    setIsVisible(true);
    setCountdown(retryAfter);
  };

  // Enhanced fetch interceptor
  useEffect(() => {
    const originalFetch = window.fetch;
    
    window.fetch = async (...args) => {
      const url = args[0] as string;
      
      try {
        const response = await originalFetch(...args);
        
        if (response.status === 429) {
          let retryAfter = 60;
          
          try {
            const retryHeader = response.headers.get('Retry-After');
            if (retryHeader) {
              retryAfter = parseInt(retryHeader);
            } else {
              const errorData = await response.clone().json();
              retryAfter = errorData.retry_after || errorData.retryAfter || 60;
            }
          } catch (e) {
            // Use default
          }
          
          showRateLimitUI(retryAfter);
        }
        
        return response;
      } catch (error) {
        throw error;
      }
    };

    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  if (!isVisible || !rateLimitInfo) {
    return null;
  }

  const progressPercent = countdown > 0 ? ((countdown / (rateLimitInfo.retryAfter || 60)) * 100) : 0;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[99999] p-4 animate-in slide-in-from-bottom duration-300">
      <div className="max-w-lg mx-auto">
        {/* Main notification card */}
        <div className="bg-gradient-to-r from-red-500 via-red-600 to-red-700 rounded-xl shadow-2xl border border-red-400 overflow-hidden">
          {/* Progress bar at top */}
          <div className="h-1 bg-red-800/30 relative">
            <div 
              className="h-full bg-red-200 transition-all duration-1000 ease-linear"
              style={{ width: `${100 - progressPercent}%` }}
            />
          </div>
          
          {/* Content */}
          <div className="p-4 text-white">
            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm">
                  <AlertCircle className="h-6 w-6 text-white animate-pulse" />
                </div>
              </div>
              
              {/* Main content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="text-lg font-bold text-white">
                    Rate Limited
                  </h3>
                  <div className="px-2 py-0.5 bg-white/20 rounded-full text-xs font-medium">
                    429
                  </div>
                </div>
                
                <p className="text-red-100 text-sm mb-3 leading-relaxed">
                  You're sending requests too quickly. Take a breather and try again in a moment.
                </p>
                
                {/* Countdown section */}
                <div className="flex items-center gap-3 p-3 bg-white/10 rounded-lg backdrop-blur-sm border border-white/20">
                  <Timer className="h-5 w-5 text-red-200 flex-shrink-0" />
                  <div className="flex-1">
                    {countdown > 0 ? (
                      <div className="text-white font-semibold">
                        Wait <span className="font-mono text-lg text-red-100">{countdown}s</span>
                      </div>
                    ) : (
                      <div className="text-green-200 font-semibold">
                        ✓ Ready to continue
                      </div>
                    )}
                    <div className="text-red-200 text-xs mt-0.5">
                      Limit: {rateLimitInfo.limit} requests per {rateLimitInfo.window}s
                    </div>
                  </div>
                  
                  {/* Circular progress */}
                  <div className="relative w-12 h-12">
                    <svg className="w-12 h-12 transform -rotate-90" viewBox="0 0 36 36">
                      <path
                        className="text-white/20"
                        strokeWidth="3"
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <path
                        className="text-white"
                        strokeWidth="3"
                        strokeDasharray={`${100 - progressPercent}, 100`}
                        strokeLinecap="round"
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        style={{ transition: 'stroke-dasharray 1s ease-in-out' }}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Clock className="h-4 w-4 text-white" />
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Close button */}
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => setIsVisible(false)}
                className="flex-shrink-0 h-8 w-8 text-white/70 hover:text-white hover:bg-white/20 transition-all duration-200"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
        
        {/* Action buttons */}
        <div className="flex justify-center mt-3">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setIsVisible(false)}
            className="bg-white/10 backdrop-blur-sm border-white/20 text-white hover:bg-white/20 transition-all duration-200"
          >
            Got it
          </Button>
        </div>
      </div>
    </div>
  );
}