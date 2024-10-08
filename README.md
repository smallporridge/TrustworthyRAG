# Trustworthiness in Retrieval-Augmented Generation Systems: A Survey

This repository contains the code for the paper:
[Trustworthiness in Retrieval-Augmented Generation Systems: A Survey](url)

![framework](framework.jpg)
We identify six essential dimensions of trustworthiness in a RAG system: 

+ **Factuality** - refers to the accuracy and truthfulness of the information generated.

+ **Transparency** - involves making the processes and decisions of the system clear and understandable to users.

+ **Accountability** - refers to the mechanisms that hold the system responsible for its actions and outputs.

+ **Privacy** - ensures the protection of personal data and user privacy.

+ **Fairness** - involves implementing strategies to minimize bias and ensure equitable treatment of all users.

+ **Robustness** - refers to the system's reliability in resisting errors and external threats.

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

The evalution results on 10 different LLMs across six dimensions are as follows:

![ability](ability.jpg)

### Citation

If you find this repo useful, please consider citing our work:

```
@inproceedings{zhou2024TrustworthyRAG,
author    = {Yujia Zhou and Yan Liu and Xiaoxi Li and Jiajie Jin and Hongjin Qian and Zheng Liu and Chaozhuo Li and Zhicheng Dou and Tsung-Yi Ho and and Philip S. Yu},
  title     = {Trustworthiness in Retrieval-Augmented Generation Systems: A Survey},
  journal   = {CoRR},
  volume    = {abs/2409.10102},
  year      = {2024}
}
```
