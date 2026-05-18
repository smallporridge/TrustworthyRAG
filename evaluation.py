import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llm_generator import GeminiGenerator, Mistral3Generator, OpenaiGenerator, ClaudeGenerator, VLLMGenerator
from prompt_builder import PromptTemplate
from input_handler import (
    input_factuality,
    input_transparency,
    input_robustness,
    input_accountability,
    input_privacy,
    input_fairness,
)
from eval_handler import (
    evaluate_factuality,
    evaluate_transparency,
    evaluate_robustness,
    evaluate_accountability,
    evaluate_privacy,
    evaluate_fairness,
)

# Mapping input process functions to evaluation types
INPUT_PROCESS_FUNC = {
    "factuality": input_factuality,
    "transparency": input_transparency,
    "robustness": input_robustness,
    "accountability": input_accountability,
    "privacy": input_privacy,
    "fairness": input_fairness,
}

# Mapping evaluation functions to evaluation types
EVALUATION_FUNC = {
    "factuality": evaluate_factuality,
    "transparency": evaluate_transparency,
    "robustness": evaluate_robustness,
    "accountability": evaluate_accountability,
    "privacy": evaluate_privacy,
    "fairness": evaluate_fairness,
}

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2-13b-chat")
    parser.add_argument("--input_dir", type=str, default="/data/")
    parser.add_argument("--output_dir", type=str, default="/results/")
    parser.add_argument("--config_dir", type=str, default="/root/data/jjj/rag_survey_2026/config/")
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="factuality",
        choices=[
            "factuality",
            "transparency",
            "robustness",
            "accountability",
            "privacy",
            "fairness"
        ],
    )
    parser.add_argument(
        "--doc_num",
        type=int,
        default=10,
        help="Number of documents, used only in transparency evaluation",
    )
    parser.add_argument("--save_note", type=str, default="")
    args = parser.parse_args()

    # Load data and configuration files
    data_path = os.path.join(args.input_dir, args.evaluation_type + ".jsonl")
    print(data_path)
    with open(data_path, "r") as f:
        if ".jsonl" in data_path:
            data = [json.loads(line) for line in f.readlines()]
        else:
            data = json.load(f)

    with open(os.path.join(args.config_dir, "model2path.json"), "r") as f:
        model2path = json.load(f)
    with open(os.path.join(args.config_dir, "task_instruction.json"), "r") as f:
        task_instruction = json.load(f)
    with open(os.path.join(args.config_dir, "openai_setting.json"), "r") as f:
        openai_setting = json.load(f)

    # Set output folder and paths
    save_note = f"_{args.save_note}" if args.save_note != "" else ""
    output_folder = os.path.join(args.output_dir, args.evaluation_type + f"{save_note}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, f"{args.evaluation_type}_{args.model_name}.json"
    )
    metric_save_path = os.path.join(
        output_folder, f"{args.evaluation_type}_{args.model_name}_metric.txt"
    )

    # Generator configuration
    config = {
        "generator_model": args.model_name,
        "generator_max_input_len": 2048,
        "gpu_memory_utilization": 0.7,
        "generator_batch_size": 5,
        "device": "cuda",
        "openai_setting": openai_setting,
    }

    # Initialize appropriate generator based on the model name
    if "gpt" in args.model_name:
        base_class = OpenaiGenerator
        config["framework"] = "openai"
        config["generation_params"] = {"max_new_tokens": 256, "do_sample": False}
    elif "gemini" in args.model_name:
        base_class = GeminiGenerator
        config["framework"] = "openai"
        config["generation_params"] = {"max_output_tokens": 256, "temperature": 0.0}
    elif "claude" in args.model_name:
        base_class = ClaudeGenerator
        config["framework"] = "openai"
        config["generation_params"] = {"temperature": 0.0}
    else:
        base_class = VLLMGenerator
        config["generator_model_path"] = model2path[config["generator_model"]]
        
    base_class = VLLMGenerator
    # base_class = Mistral3Generator
    config["generation_params"] = {"max_new_tokens": 256, "do_sample": False}
    config["framework"] = "openai"
    config["generator_model_path"] = model2path[config["generator_model"]]
        
    generator = base_class(config)

    # Set prompt template for different evaluation types
    system_prompt = task_instruction[args.evaluation_type]["system"]
    user_prompt = task_instruction[args.evaluation_type]["user"]
    prompt_template = PromptTemplate(
        config, system_prompt=system_prompt, user_prompt=user_prompt
    )

    # Generate inputs for evaluation
    input_list = []
    for item in data:
        if args.evaluation_type == "transparency":
            item_prompt = INPUT_PROCESS_FUNC[args.evaluation_type](
                item, prompt_template, args.doc_num
            )
        elif args.evaluation_type == "robustness":
            item_prompt = INPUT_PROCESS_FUNC[args.evaluation_type](
                item, prompt_template, args.doc_num
            )
        else:
            item_prompt = INPUT_PROCESS_FUNC[args.evaluation_type](
                item, prompt_template
            )
        input_list.append(item_prompt)
    print("type(input_list):", type(input_list))
    print("len(input_list):", len(input_list))
    print("type(input_list[0]):", type(input_list[0]))
    print("input_list[0]:", input_list[0])
    # Generate outputs using the generator
    output_list = generator.generate(input_list)
    del generator
    print(output_list)

    # Perform evaluation
    if args.evaluation_type == "transparency":
        nli_model_path = model2path["nli"]
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        autoais_model = T5ForConditionalGeneration.from_pretrained(
            nli_model_path
        ).to(device).eval()
        autoais_tokenizer = T5Tokenizer.from_pretrained(
            nli_model_path, use_fast=False
        )
        eval_score_dict = EVALUATION_FUNC[args.evaluation_type](
            data, output_list, autoais_model, autoais_tokenizer
        )
    else:
        eval_score_dict = EVALUATION_FUNC[args.evaluation_type](data, output_list)

    # Display outputs and evaluation scores
    
    print(eval_score_dict)

    # Save evaluation results
    for idx, (item, input, output) in enumerate(zip(data, input_list, output_list)):
        item[f"{args.evaluation_type}_input"] = input
        item[f"{args.evaluation_type}_output"] = output
        for metric in eval_score_dict:
            score = eval_score_dict[metric][idx]
            item[f"{args.evaluation_type}_{metric}_score"] = score

    # Save to output files
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    with open(metric_save_path, "w") as f:
        for metric, score_list in eval_score_dict.items():
            metric_score = []
            for score in score_list:
                if score != -1:
                    metric_score.append(score)
            f.write(f"{args.evaluation_type}_{metric}: {100*np.mean(metric_score):.2f}\n")
