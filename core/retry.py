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
    logger_name: str = "RAG.Retry"
):
    """
    Generic retry decorator / wrapper supporting both sync and async functions.
    
    Args:
        retries: Maximum number of attempts.
        backoff: Base delay factor in seconds.
        jitter: Jitter parameter.
            - If bool: if True, adds random jitter from 0 to 10% of the current delay.
            - If float: adds random jitter from 0 to jitter.
            - If tuple (min, max): adds random jitter from min to max.
        transient_errors: Single exception or tuple of exceptions that are retried by default.
        is_transient_fn: Optional custom callable to evaluate if an exception is transient.
        logger_name: Name of the logger to use.
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
                        if is_transient_fn is not None:
                            is_transient = is_transient_fn(e)
                        
                        if not is_transient or attempt == retries:
                            local_logger.error(
                                f"Async operation {func_name} failed permanently after {attempt} attempts: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise e
                        
                        # Calculate delay with backoff
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
                        if is_transient_fn is not None:
                            is_transient = is_transient_fn(e)
                        
                        if not is_transient or attempt == retries:
                            local_logger.error(
                                f"Operation {func_name} failed permanently after {attempt} attempts: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise e
                        
                        # Calculate delay with backoff
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
                        time.sleep(delay)
                
                raise Exception("Retry loop failed unexpectedly")
            return wrapper
            
    return decorator
