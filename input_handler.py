def input_factuality(item, prompt_template):
    question = item["query"]
    docs = item["positive_wrong"]
    formatted_reference = "".join([f"[{i+1}] {r} \n" for i, r in enumerate(docs)])
    item_prompt = prompt_template.get_string(
        question=question, formatted_reference=formatted_reference
    )
    return item_prompt


def input_transparency(item, prompt_template, doc_num):
    question = item["question"]
    reference = item["reference"]
    citations = item["citations"]
    if doc_num < len(reference):
        used_reference = []
        for ref in reference:
            for cit in citations:
                if cit in ref:
                    used_reference.append(ref)
        for ref in reference:
            if ref not in used_reference:
                used_reference.append(ref)
                if len(used_reference) == doc_num:
                    break
        used_reference.sort()
    else:
        used_reference = reference

    formatted_reference = "\n".join(used_reference)
    item_prompt = prompt_template.get_string(
        question=question, formatted_reference=formatted_reference
    )
    return item_prompt


def input_robustness(item, prompt_template, doc_num):
    question = item["question"]
    reference = item["reference"]
    citations = item["citations"]
    if doc_num < len(reference):
        used_reference = []
        for ref in reference:
            for cit in citations:
                if cit in ref:
                    used_reference.append(ref)
        for ref in reference:
            if ref not in used_reference:
                used_reference.append(ref)
                if len(used_reference) == doc_num:
                    break
        used_reference.sort()
    else:
        used_reference = reference

    formatted_reference = "\n".join(used_reference)
    item_prompt = prompt_template.get_string(
        question=question, formatted_reference=formatted_reference
    )
    return item_prompt


def input_accountability(item, prompt_template):
    question = item["question"]
    reference = item["reference"]
    item_prompt = prompt_template.get_string(question=question, reference=reference)
    return item_prompt


def input_privacy(item, prompt_template):
    question = item["prompt"]
    item_prompt = prompt_template.get_string(question=question)
    return item_prompt


def input_fairness(item, prompt_template):
    question = item["prompt"]
    item_prompt = prompt_template.get_string(question=question)
    return item_prompt
