import pyarrow.parquet as pq

import os
import random
import re
import json
import jsonlines
from openai import OpenAI


answer_prompt = '''
### Given the question and the existing solution steps, please generate the subsequent solution steps by following the format of the existing steps.
### Do not generate steps that have already been provided.
### Your answer should strictly follow the following format.
Step 1:
Step 2:
Step 3:
...

### Put your final answer within \\boxed{{}}.
### Question: {question}
'''

score_prompt = '''
### Given the question, standard answer, and current answer, give a score for the current answer. 
### Question: {question}
### Standard Answer: {standard_answer}
### Current Answer: {current_answer}

### You only need to give the score, and you also need to provide a detailed comparison with the standard answer to give the reason for your score.
### Provide a reward score between -100 and 100 for the answer quality, using very strict standards. Do not give a full score above 95. Make sure the reward score is an integer.
### If the final answer of the current answer is incorrect, please give a lower score.
### Your answer should strictly follow the following json format. Please note that only the following JSON is provided and no additional response content is required.
{{
    "reasoning": "",
    "score": ""
}}

### Your Answer:
'''

def split_steps(text):
    pattern = r"(step\s+\d+:)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    steps = []
    for i in range(1, len(parts), 2):
        step_title = parts[i].strip()
        step_content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        steps.append(f"{step_title} {step_content}")
    
    return steps


def extract_last_json(s):
    if '}' not in s:
        return None
    
    last_brace_index = s.rindex('}')
    
    balance = 1
    for i in range(last_brace_index - 1, -1, -1):
        if s[i] == '}':
            balance += 1
        elif s[i] == '{':
            balance -= 1

        if balance == 0:
            json_str = s[i:last_brace_index + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    
    return None


def gen_full_dataset(row):
    chosen_step_reason = []
    rejected_step_reason = []
    common_step_reason = []
    initial_reason_steps = row['initial_reason_steps'].strip()

    if initial_reason_steps.startswith("Let's think step by step. Step 1:"):
        initial_reason_steps = initial_reason_steps.replace("Let's think step by step. Step 1:", "Let's think step by step. \nStep 1:")

    pattern = r"\nStep \d+:"
    matches = re.findall(pattern, initial_reason_steps)
    
    if len(matches[-1]) == len("\nStep 1:"):
        init_end = int(matches[-1][len("\nStep ")])
    else:
        init_end = int(matches[-1][len("\nStep "):len("\nStep ")+2])
    
    if len(matches) != init_end:
        initial_reason_steps = initial_reason_steps.replace("Step 16:", "Step 4:")
        init_end = 4
        matches = re.findall(pattern, initial_reason_steps)

    if init_end != 1:
        for step_i in range(1, init_end):
            step_reason = initial_reason_steps.split("\nStep {}:".format(step_i))[1].split("\nStep {}:".format(step_i + 1))[0]
            if len(step_reason) != len(initial_reason_steps):
                common_step_reason.append(step_reason.strip())
            else:
                print('%' * 20)
                print(initial_reason_steps)
    if len(common_step_reason) != init_end - 1:
        print('$' * 20)
        print(initial_reason_steps)

    full_chosen = row['full_chosen']
    step_i = init_end
    while True:
        if "\nStep {}:".format(step_i + 1) in full_chosen:
            step_reason, full_chosen = full_chosen.split("\nStep {}:".format(step_i + 1), 1)
            chosen_step_reason.append(step_reason.strip())
        else:
            chosen_matches = re.findall(pattern, full_chosen)
            if len(chosen_matches) != 0:
                chosen_step_reason.append(full_chosen.split(chosen_matches[0], 1)[0])
            else:
                chosen_step_reason.append(full_chosen)
            break
        step_i += 1
    
    full_rejected = row['full_rejected']
    step_i = init_end
    while True:
        if "\nStep {}:".format(step_i + 1) in full_rejected:
            step_reason, full_rejected = full_rejected.split("\nStep {}:".format(step_i + 1), 1)
            rejected_step_reason.append(step_reason.strip())
        else:
            rejected_matches = re.findall(pattern, full_rejected)
            if len(rejected_matches) != 0:
                rejected_step_reason.append(full_rejected.split(rejected_matches[0], 1)[0])
            else:
                rejected_step_reason.append(full_rejected)
            break
        step_i += 1

    prompt = row['prompt']
    answer = row['answer']
    dataset = row['dataset']
    init_prompt = "Let's think step by step.\n"

    init_steps, chosen_steps, rejected_steps = [], [], []
    for step_index, step in enumerate(common_step_reason):
        init_steps += ["Step {}: {}\n".format(step_index + 1, step)]
    for chosen_index, step in enumerate(chosen_step_reason):
        chosen_steps += ["Step {}: {}\n".format(chosen_index + len(common_step_reason) + 1, step)]
    for rejected_index, step in enumerate(rejected_step_reason):
        rejected_steps += ["Step {}: {}\n".format(rejected_index + len(common_step_reason) + 1, step)]

    return prompt, answer, dataset, init_prompt, init_steps + chosen_steps, init_steps + rejected_steps


from func_timeout import func_set_timeout
@func_set_timeout(60)
def predict_model(prompt):
    messages = [{"role": "system", "content": "You are a data science expert."},
                {"role": "user", "content": prompt}]
    client = OpenAI(api_key="XXXXXXXXXXXXXXXXXXXXXXX")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        temperature=0.8
    )
    return response.choices[0].message.content


def generate_tree_data(question, init_prompt, standard_steps, error_steps, num_generated):
    T = len(standard_steps)
    tree_data = [{"response": standard_steps, "reasoning": "true answer", "score": 100}]
    
    candidate_indices = list(range(T))
    random.shuffle(candidate_indices)
    
    while len(candidate_indices) < num_generated:
        candidate_indices += random.sample(range(T), min(T, num_generated - len(candidate_indices)))

    for i in range(num_generated):
        start_idx = candidate_indices[i]
        current_steps = standard_steps[:start_idx]
        
        predicted_steps = predict_model(answer_prompt.format(question=question + '\n' + init_prompt + "".join(current_steps)))
        
        new_steps = current_steps + split_steps(predicted_steps)

        score_string = predict_model(score_prompt.format(question=question, standard_answer="".join(standard_steps), current_answer="".join(new_steps)))
        score_json = extract_last_json(score_string.replace('\\', '\\\\'))
        tree_data.append({"response": new_steps, "reasoning": score_json['reasoning'], "score": score_json['score']})
    
    score_string = predict_model(score_prompt.format(question=question, standard_answer="".join(standard_steps), current_answer="".join(error_steps)))
    score_json = extract_last_json(score_string.replace('\\', '\\\\'))
    tree_data.append({"response": error_steps, "reasoning": score_json['reasoning'], "score": score_json['score']})
        
    return tree_data


if __name__ == "__main__":
    SEED = 2024

    step_dpo_data_path = '<Step-DPO-path>/train-00000-of-00001.parquet'  # step-dpo的数据路径
    tpo_data_path = 'TPO_dataset.jsonl'

    parquet_file = pq.ParquetFile(step_dpo_data_path)
    data = parquet_file.read().to_pandas()

    # full_rejected            Find the dates when the shoe store has a sale...
    # initial_reason_steps                  Let's think step by step. \nStep 1:
    # prompt                  In the month of July, the bookstore has a sale...
    # chosen                   Find the dates on which both stores have sale...
    # full_chosen              Find the dates on which both stores have sale...
    # rejected                 Find the dates when the shoe store has a sale...
    # dataset                                                    MATH_Rephrased
    # answer                                                                  1

    for row_index, row in data.iterrows():
        prompt, answer, dataset, init_prompt, chosen_steps, rejected_steps = gen_full_dataset(row=row)
        tree_data = generate_tree_data(prompt, init_prompt, chosen_steps, rejected_steps, num_generated=3)

        data_item = {
            "index": row_index,
            "prompt": prompt + "\n" + init_prompt,
            "response_list": tree_data,
            "dataset": row["dataset"],
            "answer": row["answer"]
        }
        with jsonlines.open(tpo_data_path, mode="a") as file_jsonl:
            file_jsonl.write(data_item)
