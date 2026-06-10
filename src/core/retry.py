"""
Retry Handler - Automatic Retry for Temporary Failures

When we call external services (the AI, the database), they might fail temporarily:
- Network timeouts
- Rate limiting (too many requests)
- Service temporarily unavailable

This module automatically retries those operations with exponential backoff, giving
them time to recover before giving up. It prevents temporary issues from breaking
the user experience.
"""

import time
import random
import logging
import functools
import inspect
from typing import Callable, Any, Optional, Union, Tuple, Type

logger = logging.getLogger("RAG.Retry")


def retry(
    retries: int = 5,
    backoff: float = 1.0,
    jitter: Union[bool, float, Tuple[float, float]] = True,
    transient_errors: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]] = None,
    is_transient_fn: Optional[Callable[[BaseException], bool]] = None,
    logger_name: str = "RAG.Retry",
    on_retry_fn: Optional[Callable[[int, float, BaseException], Any]] = None,
):
    """
    Automatically retry operations that fail temporarily.
    
    This handles both sync and async functions. When an operation fails, it waits
    a bit longer before each retry (exponential backoff), with some random jitter
    to avoid overwhelming the service.
    
    Args:
        retries: How many times to try before giving up
        backoff: Base wait time in seconds (doubles each retry)
        jitter: Adds randomness to wait times to prevent thundering herd
        transient_errors: Which errors should trigger a retry
        is_transient_fn: Custom function to decide if an error is temporary
        logger_name: Where to log retry messages
        on_retry_fn: Optional callback function triggered before sleeping on retry attempts
    """
    local_logger = logging.getLogger(logger_name)
    
    # Compile exception list
    error_types = transient_errors if transient_errors is not None else (Exception,)
    if not isinstance(error_types, tuple):
        error_types = (error_types,)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Handle async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                func_name = getattr(func, "__name__", "operation")
                for attempt in range(1, retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except error_types as e:
                        # Determine transience
                        is_transient = True
                        custom_delay = None
                        if is_transient_fn is not None:
                            res = is_transient_fn(e)
                            if isinstance(res, tuple) and len(res) == 2:
                                is_transient, custom_delay = res
                            else:
                                is_transient = res
                        
                        if not is_transient or attempt == retries:
                            local_logger.error(
                                f"Async operation {func_name} failed permanently after {attempt} attempts: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise e
                        
                        # Calculate delay
                        if custom_delay is not None:
                            delay = custom_delay
                        else:
                            delay = backoff * (2 ** (attempt - 1))
                            # Apply jitter
                            if isinstance(jitter, bool):
                                if jitter:
                                    delay += random.uniform(0, 0.1 * delay)
                            elif isinstance(jitter, (int, float)):
                                delay += random.uniform(0, jitter)
                            elif isinstance(jitter, tuple) and len(jitter) == 2:
                                delay += random.uniform(jitter[0], jitter[1])
                        
                        local_logger.warning(
                            f"Async operation {func_name} failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f}s (Attempt {attempt}/{retries})...."
                        )
                        
                        if on_retry_fn is not None:
                            try:
                                if inspect.iscoroutinefunction(on_retry_fn):
                                    await on_retry_fn(attempt, delay, e)
                                else:
                                    on_retry_fn(attempt, delay, e)
                            except Exception as cb_err:
                                local_logger.error(f"Error in on_retry_fn callback: {cb_err}")
                                
                        import asyncio
                        await asyncio.sleep(delay)
                
                raise Exception("Retry loop failed unexpectedly")
            return async_wrapper
        else:
            # Handle sync functions
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                func_name = getattr(func, "__name__", "operation")
                for attempt in range(1, retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except error_types as e:
                        # Determine transience
                        is_transient = True
                        custom_delay = None
                        if is_transient_fn is not None:
                            res = is_transient_fn(e)
                            if isinstance(res, tuple) and len(res) == 2:
                                is_transient, custom_delay = res
                            else:
                                is_transient = res
                        
                        if not is_transient or attempt == retries:
                            local_logger.error(
                                f"Operation {func_name} failed permanently after {attempt} attempts: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise e
                        
                        # Calculate delay
                        if custom_delay is not None:
                            delay = custom_delay
                        else:
                            delay = backoff * (2 ** (attempt - 1))
                            # Apply jitter
                            if isinstance(jitter, bool):
                                if jitter:
                                    delay += random.uniform(0, 0.1 * delay)
                            elif isinstance(jitter, (int, float)):
                                delay += random.uniform(0, jitter)
                            elif isinstance(jitter, tuple) and len(jitter) == 2:
                                delay += random.uniform(jitter[0], jitter[1])
                        
                        local_logger.warning(
                            f"Operation {func_name} failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f}s (Attempt {attempt}/{retries})...."
                        )
                        
                        if on_retry_fn is not None:
                            try:
                                on_retry_fn(attempt, delay, e)
                            except Exception as cb_err:
                                local_logger.error(f"Error in on_retry_fn callback: {cb_err}")
                                
                        time.sleep(delay)
                
                raise Exception("Retry loop failed unexpectedly")
            return wrapper
            
    return decorator
