# new_data_tool.py
from typing import Any, List, Dict, Optional, Tuple
import re
import os
import io
import traceback
import contextlib
import pandas as pd
from pydantic import BaseModel
from .base_tool import Tool
from agentpro.model import create_model

class DataTool(Tool):
    name: str = "Data Tool"
    description: str = (
        "Analyze and manipulate CSV datasets using pandas."
        "Supports schema inspection, code generation via LLM, and execution."
    )
    action_type: str = "data_analysis"
    input_format: str = (
        "JSON with 'prompt', and either 'csv_paths' (list of file paths) "
        "or 'csv_dir' (directory containing CSV files)."
    )
    client: Any = None
    def __init__(self, provider: str = 'openai', model_name: str = 'gpt-4o-mini', api_key: str = None, litellm_provider: str = None, temperature: float = 0.1, max_tokens: Optional[int] = None,**kwargs):
        super().__init__(**kwargs)
        self.client = create_model(provider=provider, model_name=model_name, api_key=api_key, litellm_provider=litellm_provider, temperature=temperature, max_tokens=max_tokens)
    def _extract_paths(self, prompt: str) -> List[str]:
        pattern = r"[\w\-\/\\]+\.csv"
        matches = re.findall(pattern, prompt)
        return [p for p in matches if os.path.exists(p)]
    def _load_schemas(self, paths: List[str], sample: int = 5) -> Tuple[str, Dict[str, pd.DataFrame]]:
        schema_text = ""
        dfs: Dict[str, pd.DataFrame] = {}
        for path in paths:
            name = os.path.basename(path)
            var = f"df_{os.path.splitext(name)[0].replace('-', '_').replace(' ', '_')}"
            try:
                df = pd.read_csv(path)
                buf = io.StringIO()
                df.info(buf=buf)
                sample_table = df.head(sample).to_markdown(index=False)
                schema_text += (
                    f"📄 {name} as `{var}`\n"
                    f"Columns & Types:\n```\n{buf.getvalue()}```\n"
                    f"Sample Rows:\n{sample_table}\n---\n"
                )
                dfs[var] = df
            except Exception as e:
                schema_text += f"⚠️ Failed to load {name}: {e}\n"
        if not dfs:
            raise FileNotFoundError("No CSV files loaded.")
        return schema_text, dfs
    def _build_system(self) -> str:
        return (
            "You are a Python data analyst. DataFrames are pre-loaded as variables. "
            "Write pandas code only, ending with a print() of the result."
        )
    def _build_user(self, schema: str, task: str) -> str:
        return f"Available DataFrames and schema:\n{schema}\nTask: {task}\n"
    def _extract_code(self, text: str) -> Optional[str]:
        match = re.search(r"```python([\s\S]*?)```", text)
        return match.group(1).strip() if match else None
    def _execute(self, code: str, dfs: Dict[str, pd.DataFrame]) -> str:
        scope = {**dfs, 'pd': pd}
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                lines = code.strip().splitlines()
                if lines and not lines[-1].strip().startswith('print'):
                    body, last = lines[:-1], lines[-1]
                    exec("\n".join(body), {}, scope)
                    result = eval(last, {}, scope)
                    print(result)
                else:
                    exec(code, {}, scope)
        except Exception:
            return f"Execution error:\n{traceback.format_exc()}"
        return f"Output:\n{output.getvalue()}"
    def run(self, input_text: Any) -> str:
        data = input_text
        if isinstance(input_text, str):
            try:
                import json
                data = json.loads(input_text)
            except Exception:
                return "❌ Input must be JSON with 'prompt' and CSV paths."
        prompt = data.get('prompt')
        paths = data.get('csv_paths') or []
        if not paths:
            dir_ = data.get('csv_dir')
            if dir_ and os.path.isdir(dir_):
                paths = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith('.csv')]
        if not paths and prompt:
            paths = self._extract_paths(prompt)
        if not prompt or not paths:
            return "❌ Provide 'prompt' plus 'csv_paths' or 'csv_dir'."
        try:
            schema, dfs = self._load_schemas(paths)
            system = self._build_system()
            user = self._build_user(schema, prompt)
            llm_response = self.client.chat_completion(
                system_prompt=system,
                user_prompt=user
            )
            code = self._extract_code(llm_response)
            print(f"Generated code:\n{code}")
            if not code:
                return f"❌ Could not parse code from LLM:\n{llm_response}"
            return self._execute(code, dfs)
        except Exception as e:
            return f"❌ Tool error: {e}"