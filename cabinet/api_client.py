import os
import time
import random
import requests
import json


class LLMAPIError(Exception):
    pass


class ModelNotFoundError(LLMAPIError):
    def __init__(self, model: str, message: str = "Model not found"):
        super().__init__(f"{message}: {model}")
        self.model = model


# Ensure LLMFOUNDRY_TOKEN is in os.environ
def call_llm_api(model_name, messages):
    base = os.environ.get("LLMFOUNDRY_BASE_URL", "https://llmfoundry.straive.com/openai/v1")
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('LLMFOUNDRY_TOKEN')}:my-test-project",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7
    }
    max_retries = int(os.environ.get("CABINET_API_MAX_RETRIES", "5"))
    base_backoff = float(os.environ.get("CABINET_API_BACKOFF", "1.0"))

    attempt = 0
    while True:
        response = None
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.HTTPError as e:
            status = response.status_code if response is not None else None
            # 404 => model not found (do not retry)
            if status == 404:
                msg = None
                try:
                    if response.headers.get('Content-Type','').startswith('application/json'):
                        msg = response.json().get('error', {}).get('message')
                except Exception:
                    pass
                raise ModelNotFoundError(model_name, message=msg or "Model not found")

            # 429 and 5xx => retry with backoff
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                attempt += 1
                retry_after = 0.0
                if response is not None:
                    ra = response.headers.get("Retry-After")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = 0.0
                delay = max(retry_after, base_backoff * (2 ** (attempt - 1)))
                delay = delay + random.uniform(0, 0.5)
                time.sleep(delay)
                continue

            # otherwise, raise a general error
            try:
                text = response.text if response is not None else str(e)
            except Exception:
                text = str(e)
            raise LLMAPIError(text)
        except requests.RequestException as e:
            # network errors: retry a few times
            if attempt < max_retries:
                attempt += 1
                delay = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            raise LLMAPIError(str(e))
