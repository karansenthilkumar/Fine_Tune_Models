
def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

def preprocess(dataset):
    return dataset.map(format_prompt)

