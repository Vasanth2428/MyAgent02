import contextvars
import logging
import logging.handlers
import os
import builtins
import sys

_original_print = builtins.print

def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        file = kwargs.get('file', sys.stdout)
        if file is None:
            file = sys.stdout
        encoding = getattr(file, 'encoding', None) or 'utf-8'
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        
        # Convert all arguments to string and join them
        text = sep.join(map(str, args))
        try:
            # Encode with replace and decode back to string
            safe_text = text.encode(encoding, errors='replace').decode(encoding)
            new_kwargs = kwargs.copy()
            new_kwargs.pop('sep', None)
            new_kwargs.pop('end', None)
            _original_print(safe_text, sep='', end=end, **new_kwargs)
        except Exception:
            # Fallback to ascii replacement if stream encoding fails
            try:
                safe_text = text.encode('ascii', errors='replace').decode('ascii')
                new_kwargs = kwargs.copy()
                new_kwargs.pop('sep', None)
                new_kwargs.pop('end', None)
                _original_print(safe_text, sep='', end=end, **new_kwargs)
            except Exception:
                pass

builtins.print = safe_print


session_id_var = contextvars.ContextVar("session_id", default=None)

class SessionFileHandler(logging.Handler):
    """
    A logging handler that writes logs to a master file and splits logs by session_id.
    Logs for a specific session are written to logs/sessions/{session_id}.log.
    All logs are also routed to the master log file (e.g. logs/rag_engine.log).
    """
    def __init__(self, default_log_path="logs/rag_engine.log", log_dir="logs/sessions", formatter=None):
        super().__init__()
        self.default_log_path = default_log_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Dictionary to store session-specific handlers to avoid opening/closing files constantly
        self._session_handlers = {}
        
        # Initialize default master file handler
        self.master_handler = logging.handlers.RotatingFileHandler(
            self.default_log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        if formatter:
            self.master_handler.setFormatter(formatter)
            self.setFormatter(formatter)

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        self.master_handler.setFormatter(fmt)
        for h in self._session_handlers.values():
            h.setFormatter(fmt)

    def get_handler_for_session(self, session_id):
        if not session_id:
            return None
        
        if session_id not in self._session_handlers:
            # Sanitize session_id to prevent directory traversal
            safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
            if not safe_session_id:
                safe_session_id = "unknown"
            log_path = os.path.join(self.log_dir, f"{safe_session_id}.log")
            h = logging.FileHandler(log_path, encoding='utf-8')
            if self.formatter:
                h.setFormatter(self.formatter)
            self._session_handlers[session_id] = h
        return self._session_handlers[session_id]

    def emit(self, record):
        try:
            # 1. Always write to master log file
            self.master_handler.emit(record)
            
            # 2. If a session is active, write to the session-specific log file
            session_id = session_id_var.get()
            if session_id:
                handler = self.get_handler_for_session(session_id)
                if handler:
                    handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        self.master_handler.close()
        for h in self._session_handlers.values():
            h.close()
        super().close()
