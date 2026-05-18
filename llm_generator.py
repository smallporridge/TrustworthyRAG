from email.mime import text
import os
from typing import List
from urllib import response
import warnings
from copy import deepcopy
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI



class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)
        # self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def get_response(self, input: List, **params):
        response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
        return response.choices[0]

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)

        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1:
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_completion_tokens"] = max_tokens
        else:
            generation_params["max_completion_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        scores = []
        response_text = []
        for res in result:
            response_text.append(res.message.content)
            
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text

from google import genai
from google.genai import types

class GeminiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            # self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = genai.Client(
                api_key=self.openai_setting["api_key"],
                vertexai=True,
                http_options={
                "base_url": "https://api.openai-proxy.org/google"
                },
            )

    async def get_response(self, input, **params):
        def messages_to_gemini(messages):
            system_parts = []
            contents = []

            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if not content:
                    continue

                if role == "system":
                    system_parts.append(content)
                    continue

                gemini_role = "user" if role == "assistant" else "user"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })

            system_instruction = "\n".join(system_parts).strip()
            return system_instruction, contents

        system_instruction, contents = messages_to_gemini(input)

        config = {}
        if "max_tokens" in params:
            config["max_output_tokens"] = params["max_tokens"]
        if "temperature" in params:
            config["temperature"] = params["temperature"]
        if system_instruction:
            config["system_instruction"] = system_instruction
            
        # print(self.model_name)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        return response.text

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)
            print(all_result)

        return all_result


    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        if len(input_list) == 1:
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # if max_tokens is None:
        #     max_tokens = generation_params.get("max_tokens", None)
        # if max_tokens is not None:
        #     generation_params["max_output_tokens"] = max_tokens

        if return_scores:
            generation_params["logprobs"] = True

        generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        scores = []
        response_text = []

        for res in result:
            # print(f"Gemini response: {res}")
            response_text.append(res)

            if return_scores and getattr(res, "logprobs", None) is not None:
                score = np.exp([x.logprob for x in res.logprobs.content])
                scores.append(score)

        return (response_text, scores) if return_scores else response_text


from anthropic import Anthropic
from anthropic import AsyncAnthropic

class ClaudeGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            # self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncAnthropic(
                api_key=self.openai_setting["api_key"],
                base_url =  "https://api.openai-proxy.org/anthropic"
    
            )
        

    async def get_response(self, input, **params):
        def messages_to_gemini(messages):
            system_parts = []
            contents = []

            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if not content:
                    continue

                if role == "system":
                    system_parts.append(content)
                    continue

                gemini_role = "user" if role == "assistant" else "user"
                contents.append({
                    "role": gemini_role,
                    "content": [{
                        "type": "text",
                        "text": content}]
                })

            system_instruction = "\n".join(system_parts).strip()
            return system_instruction, contents

        system_instruction, contents = messages_to_gemini(input)
            
        # print(self.model_name)

        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=256,
            system = system_instruction,
            messages=contents,
        )
        
        return response.content[0].text

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)
            
            print(all_result)

        return all_result


    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        if len(input_list) == 1:
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # if max_tokens is None:
        #     max_tokens = generation_params.get("max_tokens", None)
        # if max_tokens is not None:
        #     generation_params["max_output_tokens"] = max_tokens

        if return_scores:
            generation_params["logprobs"] = True

        generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        scores = []
        response_text = []

        for res in result:
            # print(f"Gemini response: {res}")
            response_text.append(res)

            if return_scores and getattr(res, "logprobs", None) is not None:
                score = np.exp([x.logprob for x in res.logprobs.content])
                scores.append(score)

        return (response_text, scores) if return_scores else response_text



class QwenGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]


        self.openai_setting["api_key"] = "http://localhost:8000/v1"
        self.openai_setting["OPENAI_API_KEY"] = "EMPTY"

        
        self.client = AsyncOpenAI(**self.openai_setting)
        # self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def get_response(self, input: List, **params):
        response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
        return response.choices[0].message.content

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)
            print(all_result)
        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1:
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_completion_tokens"] = max_tokens
        else:
            generation_params["max_completion_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        scores = []
        response_text = []
        for res in result:
            response_text.append(res)
            
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text




class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = torch.cuda.device_count()

        self.generation_params = config["generation_params"]

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


def format_ministral_chat(messages):
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            text += f"System: {content}\n\n"
        elif role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n\n"

    text += "Assistant:"
    return text


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM

        if "gpu_memory_utilization" not in config:
            gpu_memory_utilization = 0.85
        else:
            gpu_memory_utilization = config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            tensor_parallel_size = self.gpu_num - 1
        else:
            tensor_parallel_size = self.gpu_num

        self.lora_path = None if "generator_lora_path" not in config else config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        if self.use_lora:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=True,
                max_lora_rank=64,
                max_logprobs=32016,
            )
        else:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_logprobs=32016,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    @torch.inference_mode(mode=True)
    def generate(self, input_list: List[str], return_raw_output=False, return_scores=False, **params):
        from vllm import SamplingParams

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            if "stop_token_ids" not in generation_params:
                generation_params["stop_token_ids"] = [self.tokenizer.eos_token_id]
            
        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100
                
                
        sampling_params = SamplingParams(**generation_params)
        
        normalized_inputs = []
        for i, item in enumerate(input_list):
            if isinstance(item, str):
                normalized_inputs.append(item)
            elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                # prompt = format_ministral_chat(item)
                prompt = self.tokenizer.apply_chat_template(
                    item,
                    tokenize=False,
                    add_generation_prompt=True,
                    # enable_thinking=False
                )
                normalized_inputs.append(prompt)
            else:
                raise TypeError(
                    f"input_list[{i}] has unsupported type: {type(item)}, value={item}"
                )


        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                normalized_inputs,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(normalized_inputs, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [output.outputs[0].text for output in outputs]
            base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append([np.exp(list(score_dict.values())[0].logprob) for score_dict in logprobs])
            return base_output, scores
        else:
            return base_output
        
from openai import OpenAI
from copy import deepcopy
from typing import List
import numpy as np


class Mistral3Generator(BaseGenerator):
    """Generator using vLLM OpenAI-compatible API"""

    def __init__(self, config):
        super().__init__(config)

        self.api_key = config.get("api_key", "EMPTY")
        self.api_base = config.get("api_base", "http://localhost:8000/v1")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # 自动获取模型名
        models = self.client.models.list()
        self.model_name = models.data[0].id

        self.system_prompt = config.get("system_prompt", None)

    def generate(
        self,
        input_list: List,
        return_raw_output=False,
        return_scores=False,
        **params
    ):
        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # 统一参数
        temperature = generation_params.get("temperature", 0.0)
        max_tokens = (
            generation_params.get("max_tokens")
            or generation_params.get("max_new_tokens")
            or 256
        )

        outputs = []

        for item in input_list:

            # ---------- 构造 messages ----------
            if isinstance(item, str):
                messages = [
                    {"role": "user", "content": item}
                ]

            elif isinstance(item, list) and isinstance(item[0], dict):
                messages = item

            else:
                raise TypeError(f"Unsupported input: {item}")

            # 注入 system prompt（推荐）
            if self.system_prompt is not None:
                if not (len(messages) > 0 and messages[0]["role"] == "system"):
                    messages = [
                        {"role": "system", "content": self.system_prompt}
                    ] + messages
            
            # print("messages: ",messages)

            # ---------- 调用 vLLM ----------
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if return_raw_output:
                outputs.append(response)
            else:
                outputs.append(response.choices[0].message.content)

        if return_scores:
            raise NotImplementedError("vLLM OpenAI API does not support token logprobs in this mode.")

        return outputs
    
    