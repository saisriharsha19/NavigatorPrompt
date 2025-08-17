# prompt_api/rate_limiter.py - NEW FILE (proper Redis rate limiting)
import redis
import time
import json
from fastapi import HTTPException, Request
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class RedisRateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        """Initialize Redis rate limiter (using separate DB from Celery)"""
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis rate limiter connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for rate limiting: {e}")
            # Fallback to basic in-memory for dev if Redis fails
            self.redis_client = None
            self._memory_store = {}

    def get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for client (IP + User-Agent hash)"""
        # Use forwarded IP if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Add user agent hash for better uniqueness
        user_agent = request.headers.get("User-Agent", "")
        user_agent_hash = str(hash(user_agent))[-6:]  # Last 6 digits of hash
        
        return f"rate_limit:{client_ip}:{user_agent_hash}"

    async def is_allowed(
        self, 
        request: Request, 
        limit: int = 15, 
        window_seconds: int = 30,
        endpoint: Optional[str] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        Returns: (is_allowed, metadata)
        """
        client_id = self.get_client_identifier(request)
        
        # Add endpoint specificity if provided
        if endpoint:
            client_id = f"{client_id}:{endpoint}"
        
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        if self.redis_client:
            return await self._redis_check(client_id, current_time, window_start, limit, window_seconds)
        else:
            return self._memory_check(client_id, current_time, window_start, limit, window_seconds)

    async def _redis_check(
        self, 
        client_id: str, 
        current_time: int, 
        window_start: int, 
        limit: int, 
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting with sliding window"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Use sorted set for sliding window
            key = f"requests:{client_id}"
            
            # Remove old requests outside window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 10)  # Extra buffer
            
            results = pipe.execute()
            current_count = results[1]  # Count after cleanup
            
            # Check if over limit
            is_allowed = current_count < limit
            
            # Get metadata
            if current_count > 0:
                # Get oldest request in window
                oldest_requests = self.redis_client.zrange(key, 0, 0, withscores=True)
                oldest_time = oldest_requests[0][1] if oldest_requests else current_time
                reset_time = oldest_time + window_seconds
            else:
                reset_time = current_time + window_seconds
            
            metadata = {
                "limit": limit,
                "remaining": max(0, limit - current_count - (1 if is_allowed else 0)),
                "reset_time": reset_time,
                "retry_after": max(0, reset_time - current_time) if not is_allowed else 0
            }
            
            return is_allowed, metadata
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to allowing request if Redis fails
            return True, {"limit": limit, "remaining": limit, "reset_time": current_time + window_seconds}

    def _memory_check(
        self, 
        client_id: str, 
        current_time: int, 
        window_start: int, 
        limit: int, 
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Fallback in-memory rate limiting"""
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        if client_id not in self._memory_store:
            self._memory_store[client_id] = []
        
        # Clean old requests
        self._memory_store[client_id] = [
            req_time for req_time in self._memory_store[client_id] 
            if req_time > window_start
        ]
        
        current_count = len(self._memory_store[client_id])
        is_allowed = current_count < limit
        
        if is_allowed:
            self._memory_store[client_id].append(current_time)
        
        # Calculate reset time
        if self._memory_store[client_id]:
            oldest_time = min(self._memory_store[client_id])
            reset_time = oldest_time + window_seconds
        else:
            reset_time = current_time + window_seconds
        
        metadata = {
            "limit": limit,
            "remaining": max(0, limit - current_count - (1 if is_allowed else 0)),
            "reset_time": reset_time,
            "retry_after": max(0, reset_time - current_time) if not is_allowed else 0
        }
        
        return is_allowed, metadata

    async def get_stats(self, request: Request) -> Dict[str, Any]:
        """Get current rate limit stats for client"""
        client_id = self.get_client_identifier(request)
        
        if self.redis_client:
            try:
                key = f"requests:{client_id}"
                current_time = int(time.time())
                window_start = current_time - 60  # Default 1 minute window
                
                # Clean and count
                self.redis_client.zremrangebyscore(key, 0, window_start)
                current_count = self.redis_client.zcard(key)
                
                return {
                    "client_id": client_id,
                    "current_requests": current_count,
                    "window_start": window_start,
                    "current_time": current_time
                }
            except Exception as e:
                logger.error(f"Failed to get rate limit stats: {e}")
        
        return {"client_id": client_id, "current_requests": 0}

# Global rate limiter instance
rate_limiter = None

def get_rate_limiter() -> RedisRateLimiter:
    """Get global rate limiter instance"""
    global rate_limiter
    if rate_limiter is None:
        # Use different Redis DB than Celery (DB 1 instead of 0)
        redis_url = "redis://localhost:6379/1"
        rate_limiter = RedisRateLimiter(redis_url)
    return rate_limiter