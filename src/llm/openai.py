import base64
import copy
import json
import os
import random
import re
from mimetypes import guess_type

import cv2
import requests
from azure.identity import AzureCliCredential

from src.utils.retry import retry_with_exponential_backoff


def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"


@retry_with_exponential_backoff
def call_openai_model_with_tools(
    messages,
    endpoints,
    model_name,
    api_key: str = None,
    tools: list = [],
    image_paths: list = [],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tool_choice: str = "auto",
    return_json: bool = False,
) -> dict:
    if api_key:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        }
        endpoint = "https://api.openai.com/v1"
        url = f"{endpoint}/chat/completions"
    else:
        credential = AzureCliCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token.token,
        }
        if isinstance(endpoints, str):
            endpoint = endpoints
        elif isinstance(endpoints, list):
            endpoint = random.choice(endpoints)
        else:
            raise ValueError("Endpoints must be a string or a list of strings.")
        url = f"{endpoint}/openai/deployments/{model_name}/chat/completions?api-version=2025-03-01-preview"

    payload = {
        "model": model_name,
        "messages": copy.deepcopy(messages),
    }

    is_reasoning_model = bool(re.match(r"^o\d", model_name))
    if not is_reasoning_model:
        payload["temperature"] = temperature
    if max_tokens is not None:
        if is_reasoning_model:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
    if return_json:
        payload["response_format"] = {"type": "json_object"}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    if image_paths:
        image_data_list = [local_image_to_data_url(p) for p in image_paths]
        payload["messages"].append({"role": "user", "content": []})
        for image_data in image_data_list:
            payload["messages"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": image_data}}
            )

    response = requests.post(url, headers=headers, json=payload, timeout=600)

    if response.status_code != 200:
        raise Exception(
            f"OpenAI API returned status {response.status_code}: {response.text}"
        )

    response_data = response.json()
    message = response_data["choices"][0]["message"]

    if "tool_calls" in message:
        return message
    else:
        return {"content": message["content"].strip(), "tool_calls": None}


class AzureOpenAIEmbeddingService:
    @staticmethod
    @retry_with_exponential_backoff
    def get_embeddings(endpoints, model_name, input_text, api_key: str = None):
        if api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            }
            endpoint = "https://api.openai.com/v1"
            url = f"{endpoint}/embeddings"
        else:
            if isinstance(endpoints, str):
                endpoint = endpoints
            elif isinstance(endpoints, list):
                endpoint = random.choice(endpoints)
            else:
                raise ValueError("Endpoints must be a string or a list of strings.")
            url = f"{endpoint}/openai/deployments/{model_name}/embeddings?api-version=2023-05-15"

            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/")
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token.token,
            }

        payload = {"input": input_text, "model": model_name}
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            return response.json()["data"]
        else:
            response.raise_for_status()


def extract_answer(message: dict) -> str | None:
    for call in message.get("tool_calls", []):
        args_json = call["function"]["arguments"]
        args = json.loads(args_json)
        if answer := args.get("answer"):
            return answer

    if content := message.get("content"):
        return content.strip()

    return None
