from contextvars import ContextVar

prompt_id_var: ContextVar[str] = ContextVar('prompt_id', default='unknown')
#We need this to properly propagate the prompt_id from the main prompt to the sub-prompt so we know which ones belong together