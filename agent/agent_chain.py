# pitchpilot/agent/agent_chain.py
# import os
import json
import time
from typing import List, Dict, Any, Optional

# import requests
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from .prompts import extract_prompt, market_research_prompt, business_model_prompt, slides_prompt

load_dotenv()

# HF_API_KEY = os.getenv("HF_API_KEY")
# HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")  # default suggested model
# LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gpt2")


# class HuggingFaceInferenceLLM:
#     """
#     Minimal LLM wrapper that either:
#       - Calls Hugging Face Inference API if HF_API_KEY is set, or
#       - Uses local transformers text-generation pipeline if transformers & a local model are available.

#     This wrapper intentionally avoids inheriting LangChain's LLM to bypass LangChain callback/metadata issues.
#     """

#     def __init__(self, model_name: str = HF_MODEL, max_new_tokens: int = 512, temperature: float = 0.2):
        
#         self.model_name = model_name
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         # Lazy local pipeline creation
#         self._local_pipe = None

#     def _call(self, prompt: str) -> str:
#         # Prefer HF Inference API when key is present
#         if HF_API_KEY:
#             headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#             payload = {
#                 "inputs": prompt,
#                 "parameters": {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
#             }
#             url = f"https://router.huggingface.co/v1/chat/completions/{self.model_name}"
#             print(f"Calling HF Inference API at {url} ...")
#             resp = requests.post(url, headers=headers, json=payload, timeout=60)
#             if resp.status_code == 200:
#                 j = resp.json()
#                 # response shape commonly: [{'generated_text': '...'}]
#                 if isinstance(j, list) and len(j) > 0:
#                     text = j[0].get("generated_text") or j[0].get("generated_text", "")
#                     return text
#                 # Some models return dict with 'generated_text'
#                 if isinstance(j, dict) and "generated_text" in j:
#                     return j["generated_text"]
#                 return str(j)
#             else:
#                 resp = requests.post(url, headers=headers, json=payload, timeout=60)
#             if resp.status_code == 200:
#                 j = resp.json()
#                 if isinstance(j, list) and len(j) > 0:
#                     return j[0].get("generated_text","") or str(j[0])
#                 if isinstance(j, dict) and "generated_text" in j:
#                     return j["generated_text"]
#                 return str(j)
#             elif resp.status_code == 404:
#                 # explicit remediation for 404
#                 raise RuntimeError(
#                     "HuggingFace Inference API 404: model not found. "
#                     "Check HF_MODEL environment variable and ensure the model path is correct and public, "
#                     "or that your HF_API_KEY has access to the private model. "
#                     "Example valid model path: 'google/flan-t5-large' or your-username/your-model.\n"
#                     f"Requested URL: {url}\nResponse body: {resp.text}"
#                 )
#             else:
#                 raise RuntimeError(f"HuggingFace inference failed: {resp.status_code} {resp.text}")

#         # Fallback: try local transformers pipeline
#         try:
#             # import inside method to avoid import errors if transformers not installed
#             from transformers import pipeline, set_seed
#             if self._local_pipe is None:
#                 # attempt to initialize a text-generation pipeline with the specified local model
#                 self._local_pipe = pipeline("text-generation", model=LOCAL_MODEL, device=-1)
#             set_seed(0)
#             outputs = self._local_pipe(prompt, max_length=self.max_new_tokens, do_sample=False, num_return_sequences=1)
#             return outputs[0].get("generated_text", "")
#         except Exception as e:
#             # Clear actionable message for the environment with no PyTorch/TF/Flax
#             raise RuntimeError(
#                 "No HF API key set and local transformers pipeline initialization failed. "
#                 "Either set HF_API_KEY in your .env to use Hugging Face Inference API, "
#                 "or install a supported deep-learning backend and transformers (e.g., pip install torch transformers) "
#                 f"— underlying error: {e}"
#             ) from e


import os
import requests

class HuggingFaceInferenceLLM:
    """
    Calls the Hugging Face Router chat completions endpoint:
    POST https://router.huggingface.co/v1/chat/completions
    Payload: { "model": "<owner/model[:revision]>", "messages": [...], "parameters": {...} }
    """

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, model_name: str = None, max_new_tokens: int = 512, temperature: float = 0.2):
        self.model_name = model_name or os.getenv("HF_MODEL")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # token name compatibility: prefer HF_API_KEY or HF_TOKEN
        self.api_key = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")

    def _call(self, prompt: str) -> str:
        # If user supplied a model name and an API key, call the router endpoint.
        if self.api_key and self.model_name:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model_name,
                "messages": [
                    # Optional system message can be added if you want behavior control.
                    # {"role": "system", "content": "You are a helpful assistant that outputs JSON when asked."},
                    {"role": "user", "content": prompt},
                ],
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                },
            }

            try:
                resp = requests.post(self.ROUTER_URL, headers=headers, json=payload, timeout=60)
            except requests.RequestException as e:
                raise RuntimeError(f"HuggingFace router request failed: {e}") from e

            # Handle HTTP response codes
            if resp.status_code == 200:
                j = None
                try:
                    j = resp.json()
                except Exception:
                    # return raw text fallback
                    return resp.text

                # Common router/chat response shape:
                # { "choices": [ { "message": { "role": "assistant", "content": "<text>" }, ... } ], ...}
                try:
                    # Primary: choices -> message -> content
                    choices = j.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        first = choices[0]
                        # nested message content
                        message = first.get("message") or {}
                        if isinstance(message, dict) and "content" in message:
                            # some routers use content string, or content dict with 'parts'
                            content = message["content"]
                            if isinstance(content, dict) and "text" in content:
                                return content["text"]
                            if isinstance(content, str):
                                return content
                            # if content is list (some HF formats), join parts
                            if isinstance(content, list):
                                return "".join(map(str, content))
                        # fallback: choices[0].get('text') or choices[0].get('message') as string
                        if "text" in first and isinstance(first["text"], str):
                            return first["text"]
                        # sometimes the router returns 'message' as a string
                        if isinstance(first.get("message"), str):
                            return first["message"]

                    # Fallback shapes:
                    # - Inference API older style: [{'generated_text': '...'}]
                    if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict) and "generated_text" in j[0]:
                        return j[0]["generated_text"]

                    # - simple dict with 'generated_text'
                    if isinstance(j, dict) and "generated_text" in j:
                        return j["generated_text"]

                    # - unknown but JSON, return stringified JSON as last resort
                    return str(j)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse HF router response JSON. Raw: {j}. Error: {e}") from e

            elif resp.status_code == 404:
                raise RuntimeError(
                    "Hugging Face router returned 404 Not Found: model not found or incorrect model path.\n"
                    "Action: verify HF_MODEL (self.model_name) exactly matches the model repo name on huggingface.co "
                    "and that your API token has access. Example model names: 'google/flan-t5-large' or 'your-username/your-model'.\n"
                    f"Requested model: {self.model_name}\nRouter URL: {self.ROUTER_URL}\nResponse body: {resp.text}"
                )
            else:
                # Other non-200 responses: include body for debugging
                raise RuntimeError(f"HuggingFace router inference failed: {resp.status_code} {resp.text}")

        # If no API key or no model_name, fall back to local transformers pipeline (if available).
        # Keep the failure message explicit for missing dependencies.
        try:
            from transformers import pipeline, set_seed
            local_model = os.getenv("LOCAL_MODEL", "gpt2")
            pipe = pipeline("text-generation", model=local_model, device=-1)
            set_seed(0)
            outputs = pipe(prompt, max_length=self.max_new_tokens, do_sample=False, num_return_sequences=1)
            return outputs[0].get("generated_text", outputs[0].get("text", ""))
        except Exception as e:
            raise RuntimeError(
                "No Hugging Face API key/model configured and local transformers pipeline failed. "
                "Either set HF_API_KEY/HF_TOKEN and HF_MODEL for the router endpoint, "
                "or install a DL backend and transformers (e.g., pip install torch transformers). "
                f"Underlying error: {e}"
            ) from e


class PitchAgent:
    def __init__(self):
        self.llm = HuggingFaceInferenceLLM()

    def _render_prompt(self, prompt_template: PromptTemplate, **kwargs) -> str:
                # PromptTemplate.format is safe here; LangChain PromptTemplate supports .format
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            missing = e.args[0] if e.args else "<unknown>"
            # If the missing key looks like an example block (contains commas, spaces or parentheses),
            # treat it as literal braces in the template and attempt to escape them.
            raw_template = getattr(prompt_template, "template", str(prompt_template))
            if any(ch in missing for ch in (",", " ", "(", "[", "]", ":")):
                escaped = raw_template.replace("{", "{{").replace("}", "}}")
                try:
                    return escaped.format(**kwargs)
                except Exception:
                    # Fall through to the informative error below
                    pass

            # Give an actionable message when the template contains literal braces or a missing var.
            raise RuntimeError(
                f"Prompt formatting failed: template expects variable '{missing}' which was not provided. "
                "If your prompt includes literal braces (for example example JSON, lists or CSV headers), "
                "escape them by doubling the braces (use '{{' and '}}'), or add the missing keyword when calling "
                "_call_llm_json/_render_prompt. Problematic field: " + str(missing)
            ) from e

    def _call_llm_json(self, prompt_template: PromptTemplate, **kwargs) -> Any:
        prompt_text = self._render_prompt(prompt_template, **kwargs)
        raw = self.llm._call(prompt_text)
        text = raw.strip()

        # Try to parse JSON cleanly. Many models include preface text — locate the first JSON block.
        try:
            # Fast attempt: direct parse
            return json.loads(text)
        except Exception:
            import re
            m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if m:
                candidate = m.group(1)
                try:
                    return json.loads(candidate)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON from LLM output. Extracted: {candidate}\nRaw: {text}") from e
            # If nothing parsed, raise with raw text for debugging
            raise RuntimeError(f"LLM did not return JSON. Raw output: {text}")

    def run_workflow(self, idea: str) -> List[Dict[str, str]]:
        # Step 1: Extract category, problem, solution
        extract = self._call_llm_json(extract_prompt, idea=idea)
        category = extract.get("category", "General") if isinstance(extract, dict) else "General"
        problem = extract.get("problem", "") if isinstance(extract, dict) else ""
        solution = extract.get("solution", "") if isinstance(extract, dict) else ""

        time.sleep(0.4)

        # Step 2: Market research
        market = self._call_llm_json(market_research_prompt, category=category, idea=idea)
        market_summary = {
            "market_size_estimate": market.get("market_size_estimate", "") if isinstance(market, dict) else "",
            "key_trends": market.get("key_trends", []) if isinstance(market, dict) else [],
            "top_competitors": market.get("top_competitors", []) if isinstance(market, dict) else [],
        }

        time.sleep(0.4)

        # Step 3: Suggest business model
        bm = self._call_llm_json(
            business_model_prompt,
            category=category,
            problem=problem,
            solution=solution,
            market_summary=json.dumps(market_summary, ensure_ascii=False),
        )
        business_models = bm.get("business_models", []) if isinstance(bm, dict) else []

        time.sleep(0.4)

        # Step 4: Generate slides
        slides = self._call_llm_json(
            slides_prompt,
            category=category,
            problem=problem,
            solution=solution,
            market_summary=json.dumps(market_summary, ensure_ascii=False),
            business_models=json.dumps(business_models, ensure_ascii=False),
        )

        final_slides = []
        if isinstance(slides, list):
            for s in slides:
                if isinstance(s, dict):
                    title = s.get("title", "Slide")
                    content = s.get("content", "")
                else:
                    # If model returned plain strings, attempt to split on colon
                    text = str(s)
                    if ":" in text:
                        parts = text.split(":", 1)
                        title = parts[0].strip()
                        content = parts[1].strip()
                    else:
                        title = "Slide"
                        content = text
                final_slides.append({"title": title, "content": content})
        else:
            raise RuntimeError("Slides output was not a JSON list.")

        # Guarantee 8-10 slides: pad or truncate
        if len(final_slides) < 8:
            needed = 8 - len(final_slides)
            for i in range(needed):
                final_slides.append({"title": f"Extra {i+1}", "content": problem or solution or "Details."})
        final_slides = final_slides[:10]

        return final_slides
