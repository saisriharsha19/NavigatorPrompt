# prompt_api/middleware.py - FIX the rate limiting middleware
import time
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from rate_limiter import get_rate_limiter
from config import settings
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, endpoint_overrides: Dict[str, Dict[str, int]] = None):
        super().__init__(app)
        self.rate_limiter = get_rate_limiter()
        self.enabled = settings.RATE_LIMIT_ENABLED
        
        # Default rate limits per HTTP method
        self.method_limits = {
            "GET": {"limit": 100, "window": 60},
            "POST": {"limit": 20, "window": 60},
            "PUT": {"limit": 10, "window": 60},
            "DELETE": {"limit": 5, "window": 60},
        }
        
        # Specific endpoint rate limit overrides
        self.endpoint_overrides = endpoint_overrides or {
            "/prompts/generate": {"limit": 5, "window": 60},
            "/prompts/evaluate": {"limit": 8, "window": 60},
            "/prompts/suggest-improvements": {"limit": 10, "window": 60},
            "/prompts/analyze-and-tag": {"limit": 10, "window": 60},
            "/library/prompts": {"limit": 30, "window": 60},
            "/auth/session": {"limit": 60, "window": 60},
            "/tasks": {"limit": 50, "window": 60},
            "/health": {"limit": 120, "window": 60},
        }

    def should_skip_rate_limiting(self, request: Request) -> bool:
        """Determine if this request should skip rate limiting"""
        if not self.enabled:
            return True
            
        skip_patterns = [
            "/docs", "/redoc", "/openapi.json", 
            "/favicon.ico", "/static/"
        ]
        
        path = request.url.path
        return any(path.startswith(pattern) for pattern in skip_patterns)

    def get_rate_limit_config(self, request: Request) -> Dict[str, int]:
        """Get rate limit configuration for this request"""
        path = request.url.path
        method = request.method
        
        # Check for specific endpoint overrides first
        for endpoint_pattern, limits in self.endpoint_overrides.items():
            if path.startswith(endpoint_pattern):
                return limits
        
        # Fall back to method-based limits
        return self.method_limits.get(method, {"limit": 20, "window": 60})

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Skip rate limiting if disabled or for certain paths
        if self.should_skip_rate_limiting(request):
            response = await call_next(request)
            self.add_cors_headers(response, request)
            return response
        
        try:
            # Get rate limit configuration for this request
            config = self.get_rate_limit_config(request)
            limit = config["limit"]
            window = config["window"]
            
            # Check rate limit
            is_allowed, metadata = await self.rate_limiter.is_allowed(
                request, 
                limit=limit, 
                window_seconds=window,
                endpoint=request.url.path
            )
            
            if not is_allowed:
                # Log rate limit violation
                client_id = self.rate_limiter.get_client_identifier(request)
                logger.warning(
                    f"Rate limit exceeded: {client_id} on {request.method} {request.url.path} "
                    f"(limit: {limit}/{window}s, retry_after: {metadata['retry_after']}s)"
                )
                
                # FIX: Return JSONResponse directly instead of raising HTTPException
                response = JSONResponse(
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {limit} per {window} seconds.",
                        "retry_after": metadata["retry_after"],
                        "limit": metadata["limit"],
                        "window": window
                    },
                    status_code=429
                )
                
                # Add rate limit headers
                self.add_rate_limit_headers(response, metadata, window)
                self.add_cors_headers(response, request)
                
                return response
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            self.add_rate_limit_headers(response, metadata, window)
            self.add_cors_headers(response, request)
            
            # Log slow requests for monitoring
            process_time = time.time() - start_time
            if process_time > 5:
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {process_time:.2f}s (client: {self.rate_limiter.get_client_identifier(request)})"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue with request if rate limiting fails
            response = await call_next(request)
            self.add_cors_headers(response, request)
            return response

    def add_rate_limit_headers(self, response, metadata, window):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
        response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        response.headers["X-RateLimit-Reset"] = str(metadata["reset_time"])
        response.headers["X-RateLimit-Window"] = str(window)

    def add_cors_headers(self, response, request):
        """Add CORS headers to response"""
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Expose-Headers"] = "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-RateLimit-Window"

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers for all responses
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block", 
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Permitted-Cross-Domain-Policies": "none",
        }
        
        # Add all security headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Add HSTS only for HTTPS connections
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request logging with security monitoring"""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"in {process_time:.3f}s"
        )
        
        # Log errors for monitoring
        if response.status_code >= 400:
            logger.warning(
                f"Error response: {response.status_code} for {request.method} {request.url.path} "
                f"from {client_ip}"
            )
        
        return response