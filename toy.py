import os
import time
import linecache
import functools
import sys
import torch

from typing import Callable

# Current project root directory
PROJECT_ROOT = os.path.abspath(".")

def in_project(frame):
    """Determine if the current frame belongs to project code"""
    fn = frame.f_code.co_filename
    if not fn or fn.startswith("<"):
        return False  # Exclude <frozen os>, <string>, <built-in>, etc.
    fn = frame.f_code.co_filename
    return os.path.abspath(fn).startswith(PROJECT_ROOT)

def format_seconds(seconds: float) -> str:
    # Define units
    units = ['ns', 'µs', 'ms', 's', 'min', 'hr']
    thresholds = [1e-9, 1e-6, 1e-3, 1, 60, 3600]
    
    size = seconds
    n = 0
    while n < len(thresholds) - 1 and size >= thresholds[n + 1]:
        n += 1
    if units[n] == 'min':
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes}m {remainder:.1f}s"
    elif units[n] == 'hr':
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        minutes = int(remainder // 60)
        seconds_left = remainder % 60
        return f"{hours}h {minutes}m {seconds_left:.1f}s"
    else:
        scaled = seconds / thresholds[n]
        return f"{scaled:.2f} {units[n]}"

def format_bytes(size: float) -> str:
    # Define units
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    power = 1024
    n = 0
    while size >= power and n < len(units) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {units[n]}"

def cuda_mem():
    """Return (allocated, reserved) GPU memory. If no GPU is available, return (0,0)."""
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

def cuda_timer():
    torch.cuda.synchronize()
    return time.perf_counter()

def make_cuda_tracer(target_fn_name: str | None = None):

    prev = {
        "alloc": 0,
        "resv": 0,
        "time": time.perf_counter()
    }

    def tracer(frame, event, arg):

        # Only activate call/line/return events for files within the project
        if not in_project(frame):
            return None

        if event not in ("call", "line", "return", "exception"):
            return None

        if event == "call" and target_fn_name is not None:
            return tracer if frame.f_code.co_name == target_fn_name else None

        prefix = {
            "line": "",
            "call": "CALL ",
            "return": "RETURN ",
            "exception": "EXCEPTION "
        }.get(event, "")

        fn_name = frame.f_code.co_name
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        source_line = linecache.getline(filename, lineno).strip()

        alloc, resv = cuda_mem()
        now = time.perf_counter()
        elapsed = now - prev["time"]

        delta_alloc = alloc - prev["alloc"]
        delta_resv = resv - prev["resv"]

        sign_alloc = "+" if delta_alloc >= 0 else "-"
        sign_resv = "+" if delta_resv >= 0 else "-"

        speed_alloc = delta_alloc / elapsed if elapsed > 0 else 0
        speed_resv = delta_resv / elapsed if elapsed > 0 else 0

        print(
            f"      elapsed={format_seconds(elapsed)}\n"
            f"      alloc={format_bytes(alloc)} ({sign_alloc}{format_bytes(abs(delta_alloc))}) "
            f" → {format_bytes(abs(speed_alloc))}/s\n"
            f"      reserved={format_bytes(resv)} ({sign_resv}{format_bytes(abs(delta_resv))}) "
            f" → {format_bytes(abs(speed_resv))}/s\n"
            f"{prefix}{os.path.relpath(filename, PROJECT_ROOT)}:{fn_name}:{lineno} | {source_line}"
        )

        prev["alloc"] = alloc
        prev["resv"] = resv
        prev["time"] = time.perf_counter()

        return tracer

    return tracer

def trace(tracer):
    """
    Decorator factory that applies a sys.settrace tracer to a function.
    Usage:
        @trace(my_tracer)
        def f(): ...
    """
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_tracer = sys.gettrace()
            sys.settrace(tracer)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(old_tracer)

        return wrapper

    return decorator

def cuda_trace(func):
    """
    Decorator that applies a torch CUDA tracer to a function.
    Usage:
        @cuda_trace
        def f(): ...
    """
    tracer = make_cuda_tracer()
    return trace(tracer)(func)

def conditional(cond: Callable[..., bool], decorator: Callable[[Callable], Callable] | None = None):
    """
    Conditionally applies a decorator to a function.

    Usage:
        @conditional(cond_fn)
        @decorator
        def f(...): ...

        @conditional(cond_fn, decorator)
        def f(...): ...
    """
    def conditional_decorator(func):
        decorated = decorator(func) if decorator else func
        raw = func.__wrapped__ if decorator is None else func

        @functools.wraps(func)
        def conditional_wrapper(*args, **kwargs):
            if cond(*args, **kwargs):
                return decorated(*args, **kwargs)
            return raw(*args, **kwargs)

        return conditional_wrapper

    return conditional_decorator
