import json
import os
import time
import urllib.request

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_PURIFICATION_PROMPT = """You are cleaning scientific or factual node text for graph learning.

Rewrite the input text into a concise, objective, information-dense version.

Rules:
1. Remove subjective tone, opinions, persuasion, and marketing language.
2. Remove redundant generic phrases and boilerplate.
3. Preserve core semantics, named entities, technical terms, and important relations.
4. Do not invent facts.
5. Output only the purified text.

Input text:
{text}
"""


class LLMTextPurifier:
    def __init__(self, prompt_template=None):
        self.prompt_template = prompt_template or DEFAULT_PURIFICATION_PROMPT

    def build_prompt(self, text):
        return self.prompt_template.format(text=text.strip())

    def purify_text(self, text):
        raise NotImplementedError

    def purify_texts(self, texts, show_progress=True):
        iterable = texts
        if show_progress:
            iterable = tqdm(texts, desc="Purifying texts")
        return [self.purify_text(text) for text in iterable]


class MockTextPurifier(LLMTextPurifier):
    def purify_text(self, text):
        return text.strip()


class LocalModelTextPurifier(LLMTextPurifier):
    def __init__(
        self,
        model_name,
        prompt_template=None,
        device=None,
        max_input_length=512,
        max_new_tokens=192,
        temperature=0.0,
    ):
        super().__init__(prompt_template=prompt_template)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model, self.is_seq2seq = self._load_generation_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_generation_model(self, model_name):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return model, True
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return model, False

    def _decode_generated_text(self, prompt, generated_ids):
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if self.is_seq2seq:
            return decoded
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :].strip()
        return decoded

    def purify_text(self, text):
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)

        result = self._decode_generated_text(prompt, generated[0])
        return result if result else text.strip()


class APITextPurifier(LLMTextPurifier):
    def __init__(
        self,
        model_name=None,
        prompt_template=None,
        api_url=None,
        api_key=None,
        timeout=120,
        retry_wait_seconds=5,
    ):
        super().__init__(prompt_template=prompt_template)
        self.model_name = model_name or os.getenv("TEXT_PURIFIER_API_MODEL", "")
        self.api_url = api_url or os.getenv("TEXT_PURIFIER_API_URL", "")
        self.api_key = api_key or os.getenv("TEXT_PURIFIER_API_KEY", "")
        self.timeout = timeout
        self.retry_wait_seconds = retry_wait_seconds

        if not self.api_url:
            raise ValueError(
                "API mode requires TEXT_PURIFIER_API_URL or --api_url to be provided."
            )

    def purify_text(self, text):
        payload = {
            "model": self.model_name,
            "prompt": self.build_prompt(text),
            "input_text": text,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.api_url,
            data=data,
            headers=headers,
            method="POST",
        )

        last_error = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                parsed = self._parse_api_response(response_payload)
                return parsed if parsed else text.strip()
            except Exception as exc:
                last_error = exc
                time.sleep(self.retry_wait_seconds)

        raise RuntimeError(f"API purification failed: {last_error}")

    @staticmethod
    def _parse_api_response(payload):
        if isinstance(payload, dict):
            if "output" in payload:
                return str(payload["output"]).strip()
            if "text" in payload:
                return str(payload["text"]).strip()
            if "choices" in payload and payload["choices"]:
                choice = payload["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        return str(choice["message"].get("content", "")).strip()
                    if "text" in choice:
                        return str(choice["text"]).strip()
        return ""


def build_text_purifier(
    mode,
    prompt_template=None,
    model_name=None,
    api_url=None,
    api_key=None,
):
    mode = mode.lower()
    if mode == "mock":
        return MockTextPurifier(prompt_template=prompt_template)
    if mode == "local":
        if not model_name:
            raise ValueError("Local mode requires --model_name.")
        return LocalModelTextPurifier(
            model_name=model_name,
            prompt_template=prompt_template,
        )
    if mode == "api":
        return APITextPurifier(
            model_name=model_name,
            prompt_template=prompt_template,
            api_url=api_url,
            api_key=api_key,
        )
    raise ValueError(f"Unsupported purifier mode: {mode}")
