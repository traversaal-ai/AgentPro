from .base_tool import Tool
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
import math
import requests
import json
import os

class AresInternetTool(Tool):
    name: str = "Ares Internet Search"
    description: str = "Uses Ares API to search live and detailed information from the internet and returns a clean summary and related links."
    action_type: str = "ares_internet_search"
    input_format: str = "A search query as a string. Example: 'Best restaurants in San Francisco'"

    _config: Dict[str, Any] = PrivateAttr()

    def __init__(self, api_key: Optional[str] = None, **data):
        super().__init__(**data)
        self._config = {
            "api_key": api_key or os.getenv("ARES_API_KEY")
        }

    def run(self, input_text: Any) -> str:
        if not isinstance(input_text, str):
            return "❌ Error: Expected a search query string."

        api_key = self._config.get("api_key")

        if not api_key:
            return "❌ Error: Ares API key is missing. Please provide it during initialization or set the ARES_API_KEY environment variable."

        api_url = "https://api-ares.traversaal.ai/live/predict"
        prompt = str(input_text).strip("'\"")
        payload = {"query": prompt}
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json"
        }

        # print(f"Sending payload: {payload}")

        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                return f"Error: Ares API returned {response.status_code} - {response.text}"

            result = response.json()

            response_text = result.get("data", {}).get("response_text", "").strip()
            web_urls = result.get("data", {}).get("web_url", [])

            if not response_text:
                return "No information found for this query. Please try a different search term."

            output = f"Search Summary:\n{response_text}\n\n"

            if web_urls:
                output += "Related Links:\n"
                for idx, url in enumerate(web_urls, 1):
                    output += f"{idx}. {url}\n"

            return output.strip()

        except requests.exceptions.RequestException as e:
            return f"Error: HTTP request failed - {e}"

        except Exception as e:
            return f"Error: Unexpected error - {e}"
