from contextvars import ContextVar

prompt_id_var: ContextVar[str] = ContextVar('prompt_id', default='unknown')