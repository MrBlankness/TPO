from torch.utils.data import Dataset
import json
import copy


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = file.readlines()
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        prompt = data['prompt']
        chosen = "".join(data['response_list'][0]['response'])
        rejected = "".join(data['response_list'][-1]['response'])

        chosen_full_text = f"{prompt}\n\n### Response:\n{chosen}"
        rejected_full_text = f"{prompt}\n\n### Response:\n{rejected}"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_full_tokens = self.tokenizer.encode(chosen_full_text, add_special_tokens=False)
        rejected_full_tokens = self.tokenizer.encode(rejected_full_text, add_special_tokens=False)

        input = {
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            }
        return input

    def __len__(self):
        return len(self.data_list)



class TPODataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = file.readlines()
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        prompt = data['prompt']
        response_list = [data['response_list'][i]['response'] for i in range(len(data['response_list']))]
        score_list = [data['response_list'][i]['score'] for i in range(len(data['response_list']))]

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_prefix_text = f"{prompt}\n\n### Response:\n"
        response_prefix_tokens = self.tokenizer.encode(response_prefix_text, add_special_tokens=False)

        response_full_token_list = []
        response_full_step_index_list = []
        min_step_number = 100000
        for response in response_list:
            response_full_token = response_prefix_tokens
            response_full_step_index = [0] * len(response_full_token)
            for index, step in enumerate(response):
                step_token = self.tokenizer.encode(step, add_special_tokens=False)
                step_index = [index + 1] * len(step_token)
                response_full_token += step_token
                response_full_step_index += step_index
            response_full_token_list.append(copy.deepcopy(response_full_token))
            response_full_step_index_list.append(copy.deepcopy(response_full_step_index))
            min_step_number = min(min_step_number, max(response_full_step_index))
        
        for index in range(len(response_full_step_index_list)):
            response_full_step_index_list[index] = [min(x, min_step_number) for x in response_full_step_index_list[index]]

        input = {
                "prompt": prompt_tokens,
                "response_list": response_full_token_list,
                "step_index": response_full_step_index_list,
                "score_list": score_list
            }
        return input

    def __len__(self):
        return len(self.data_list)