from torch.utils.data import DataLoader, random_split
import torch
from dataset import TPODataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loss import compute_tpo_batch_loss
from evaluate import evaluate_loss_dataloader
import time
from functools import partial
from peft import get_peft_model, LoraConfig, TaskType


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
 
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
 
 
def init_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
 
    target_modules = find_all_linear_names(model)
 
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
 
    model = model.to(device)
    return model, tokenizer


device = torch.device("cuda:0")
model_path = '<LLM-path>'   # 填写LLM的路径
# Use Lora to finetune the LLM
model, tokenizer = init_model(model_path, device)
# Full parameter fine-tuning
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
# model.to(device)
ref_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
ref_model.eval()
ref_model.to("cuda:1")


data_file = 'TPO_dataset.jsonl'
dataset = TPODataset(data_file, tokenizer)
train_size = int(len(dataset) * 0.85)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

IGNORE_INDEX = False


def data_collate(batch, pad_token_id, device, max_length=None, if_mask_prompt=True):
    batch_data = {
        "prompt": [],
        "response_list": [],
        "response_list_mask": [],
        "step_index": [],
        "score_list": []
    }

    max_length_common = 0
    for item in batch:
        for response in item["response_list"]:
            max_length_common = max(max_length_common, len(response))

    for item in batch:
        prompt = torch.tensor(item['prompt'])
        batch_data['prompt'].append(prompt)

        out_padding_list, mask_list, step_index_list = [], [], []
        for index, out in enumerate(item["response_list"]):
            out_padding = out + [pad_token_id] * (max_length_common - len(out))
            out_padding_list.append(out_padding)

            mask = torch.ones(len(out_padding)).bool()
            mask[len(item["step_index"][index]):] = IGNORE_INDEX
            if if_mask_prompt:
                mask[:prompt.shape[0] + 2] = IGNORE_INDEX
            mask_list.append(mask)

            step_index_list.append(item["step_index"][index] + [0] * (max_length_common - len(item["step_index"][index])))
        
        batch_data["response_list"].append(torch.tensor(out_padding_list))
        batch_data["response_list_mask"].append(torch.stack(mask_list))
        batch_data["step_index"].append(torch.tensor(step_index_list))
        batch_data["score_list"].append(torch.tensor(item['score_list']))


    for key in ["response_list", "response_list_mask", "step_index"]:
        tensor_stack = torch.stack(batch_data[key])
        if max_length is not None:
            tensor_stack = tensor_stack[:, :, :max_length]
        batch_data[key] = tensor_stack.to(device)
    batch_data["score_list"] = torch.stack(batch_data["score_list"]).to(device)

    return batch_data


customized_collate_fn = partial(
    data_collate,
    pad_token_id=tokenizer.pad_token_id,
    device=device,
    if_mask_prompt=True,
    max_length=1024
)

batch_size = 1
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)

def train_model(
        policy_model, reference_model, train_loader, val_loader,
        optimizer, num_epochs, beta,
        eval_freq, eval_iter):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    global_step = -1

    for epoch in range(num_epochs):
        policy_model.train()

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_tpo_batch_loss(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                res = evaluate_loss_dataloader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter,
                    method='TPO'
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

    return tracking


def main():
    torch.manual_seed(42)
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, weight_decay=0.01)

    num_epochs = 1
    tracking = train_model(
        policy_model=model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.5,
        eval_freq=2,
        eval_iter=2
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()