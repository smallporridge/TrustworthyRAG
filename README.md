# Trustworthiness in Retrieval-Augmented Generation Systems: A Survey

This repository contains the code for the paper:
[Trustworthiness in Retrieval-Augmented Generation Systems: A Survey](url)

## Quick start

### Install environment

Install all required libraries by running
```bash
pip install -r requirements.txt
```

### Setup Model Path and Openai key (Optional)

You need to fill in the local path of the LLM you are using in `/config/model2path.json`, otherwise it will be downloaded from Huggingface by default. 

If you need to use Openai APIs such as GPT-4o, you need to configure the `api_key` and other settings in the `/config/openai_setting.json`.

### Run Evaluation

Just use following code to run evaluation on dimensions. You can modify the `model_list` inside to determine the model you want to evaluate.

```bash
bash run_eval.sh
```
