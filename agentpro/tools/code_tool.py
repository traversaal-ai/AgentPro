from typing import Any, Optional, Tuple, List
import re
import subprocess
import sys
import traceback
from pydantic import PrivateAttr
from .base_tool import Tool
from agentpro.model import create_model

class CodeEngine(Tool):
    name: str = "Code Generation & Execution Tool"
    description: str = (
        "Generates Python code from a prompt and then executes generated code. "
        "Returns the code and execution results or errors."
    )
    action_type: str = "code_generation_execution"
    input_format: str = (
        "JSON with 'prompt'. Prompt should be a code description. "
        "Tool generates/extracts code, installs dependencies, and runs it." 
    )
    _client: Any = PrivateAttr()
    def __init__(
        self,
        provider: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: str = None,
        litellm_provider: str = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = create_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            litellm_provider=litellm_provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
    def _extract_code_block(self, text: str) -> Optional[str]:
        m = re.search(r"```python([\s\S]*?)```", text)
        return m.group(1).strip() if m else None
    def _install_packages(self, first_line: str) -> List[str]:
        parts = first_line.lstrip('#').strip().split()
        if len(parts) >= 3 and parts[0]=='pip' and parts[1]=='install':
            return parts[2:]
        return []
    def _execute_code(self, code: str) -> Tuple[str, Optional[str]]:
        lines = code.splitlines()
        if lines and lines[0].strip().startswith('pip install'):
            pkgs = self._install_packages(lines[0])
            for pkg in pkgs:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
            code = '\n'.join(lines[1:])
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            return code, None
        except Exception:
            tb = traceback.format_exc()
            return code, tb
    def _generate_code(self, prompt: str) -> str:
        system = (
            "You are a Python code generator. Respond only with executable code inside ```python...```. "
            "List pip installs as commented first line if needed."
        )
        user = f"Generate Python code to {prompt}."
        return self._client.chat_completion(
            system_prompt=system,
            user_prompt=user
        )
    def run(self, input_text: Any) -> str:
        data = input_text
        if isinstance(input_text, str):
            try:
                import json
                data = json.loads(input_text)
            except Exception:
                data = {'prompt': input_text}

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return "❌ Provide a non-empty 'prompt'."
        if prompt.startswith('```python') or prompt.startswith('import') or prompt.startswith('#') or 'def ' in prompt:
            code = self._extract_code_block(prompt) or prompt
        else:
            llm_resp = self._generate_code(prompt)
            code = self._extract_code_block(llm_resp) or llm_resp
        executed_code, error = self._execute_code(code)
        if error:
            return f"Code:\n{executed_code}\n\nExecution error:\n{error}"
        return f"Code:\n{executed_code}\n\nExecution successful."