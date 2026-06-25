"""
Light-weight Azure OpenAI chat wrapper
"""
from __future__ import annotations

import base64
import mimetypes
from typing import List, Optional
from openai import AzureOpenAI

from src.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    DEPLOYMENT_NAME_GPT41,
    API_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE
)


class OpenAILLM:
    def __init__(
        self,
        input_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self.endpoint = azure_endpoint or AZURE_OPENAI_ENDPOINT
        used_key = input_api_key or AZURE_OPENAI_KEY

        self.client = AzureOpenAI(
            api_key=used_key,
            azure_endpoint=self.endpoint,
            api_version=api_version or API_VERSION,
        )
        self.model = model_name or DEPLOYMENT_NAME_GPT41

    @staticmethod
    def _encode_image_to_data_url(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def chat_completion(
            self,
            messages: List[dict[str, str]],
            *,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE,
            image_path: Optional[str] = None
    ) -> str:
        """
        Thin wrapper around `client.chat.completions.create`.

        Supports optional image input.

        Returns: message content as a string.
        Raises: the original exception if the API call fails.
        """
        # If image is provided convert to multimodal format
        if image_path:
            data_url = self._encode_image_to_data_url(image_path)

            new_messages = []
            for m in messages:
                if m.get("role") == "user":
                    new_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": m.get("content", "")},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            }
                        ]
                    })
                else:
                    new_messages.append(m)

            messages_to_send = new_messages
        else:
            messages_to_send = messages

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages_to_send,
            max_tokens=max_tokens,
            temperature=temperature
        )

        content = (resp.choices[0].message.content or "").strip()

        if not content:
            raise ValueError("LLM call succeeded but returned empty content.")

        return content

    def validate_api_key(self) -> tuple[bool, str]:
        """
        Fire a trivial request to check whether the key / deployment works.
        """
        try:
            _ = self.chat_completion(messages=[{"role": "user", "content": "ping"}], max_tokens=5, temperature=0.0)
            return True, "API key is valid."
        except Exception as e:
            return False, f"Unable to verify API key: {e}"
