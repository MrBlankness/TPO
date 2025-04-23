import torch.nn.functional as F
import torch.nn as nn
import torch


def compute_step_weights(hidden_state, step_index):
    batch_size, N, n_token, hidden_dim = hidden_state.shape
    step_weights = torch.zeros(batch_size, N, step_index.max() + 1, device=hidden_state.device)

    for b in range(batch_size):
        for n in range(N):
            for step in range(1, step_index.max() + 1):
                step_mask = (step_index[b, n] == step)
                if step_mask.sum() == 0:
                    continue

                step_hidden = hidden_state[b, n][step_mask].mean(dim=0)

                ref_step_mask = (step_index[b, 0] == step)
                if ref_step_mask.sum() == 0:
                    continue
                ref_step_hidden = hidden_state[b, 0][ref_step_mask].mean(dim=0)

                cosine_similarity = F.cosine_similarity(step_hidden, ref_step_hidden, dim=0)
                step_weights[b, n, step] = cosine_similarity + 1

    return step_weights


def compute_weighted_logits(logit, step_weights, step_index):
    batch_size, N, n_token, output_dim = logit.shape
    weighted_logits = torch.zeros(batch_size, N, output_dim, device=logit.device)

    for b in range(batch_size):
        for n in range(N):
            step_logit_sum = torch.zeros(output_dim, device=logit.device)
            total_weight = 0.0

            for step in range(1, step_weights.shape[2]):
                step_mask = (step_index[b, n] == step)
                if step_mask.sum() == 0:
                    continue

                step_logit = logit[b, n][step_mask].mean(dim=0)

                weight = step_weights[b, n, step]
                step_logit_sum += weight * step_logit
                total_weight += weight

            if total_weight > 0:
                weighted_logits[b, n] = step_logit_sum / total_weight

    return weighted_logits


def lambda_rank_loss(predicted_logits, labels):
    batch_size, N = predicted_logits.shape

    diff = predicted_logits.unsqueeze(2) - predicted_logits.unsqueeze(1)
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)

    pairwise_loss = -F.logsigmoid(diff * label_diff.sign())
    loss = pairwise_loss.sum(dim=(1, 2)) / (N * (N - 1))

    return loss.mean()


class DPOLoss(nn.Module):

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
            reference_chosen_logps: torch.Tensor,
            reference_rejected_logps: torch.Tensor,
    ):
        policy_logps = policy_chosen_logps - policy_rejected_logps
        reference_logps = reference_chosen_logps - reference_rejected_logps
        logits = policy_logps - reference_logps

        loss = -F.logsigmoid(self.beta * logits)

        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class SimPo(nn.Module):

    def __init__(self, beta: float = 0.1, gamma: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
    ):
        logits = policy_chosen_logps - policy_rejected_logps
        logits = logits - self.gamma
        loss = -F.logsigmoid(self.beta * logits)

        return loss.mean()


class TPOLoss(nn.Module):

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
            self,
            policy_responses_logps: torch.Tensor,
            reference_responses_logps: torch.Tensor,
            hidden_state: torch.Tensor,
            step_index: torch.Tensor,
            labels: torch.Tensor
    ):
        logits = policy_responses_logps - reference_responses_logps

        step_weights = compute_step_weights(hidden_state, step_index)
        weighted_logits = compute_weighted_logits(logits, step_weights, step_index)
        text_logits = weighted_logits.mean(dim=-1)
        loss = -F.logsigmoid(self.beta * lambda_rank_loss(text_logits, labels))

        chosen_rewards = (policy_responses_logps[:, 0, :] - reference_responses_logps[:, 0, :]).detach()
        rejected_rewards = (policy_responses_logps[:, -1, :] - reference_responses_logps[:, -1, :]).detach()

        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_dpo_logprobs(logits, labels, mask=None):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    logps = F.log_softmax(logits, dim=-1)

    select_logprobs = torch.gather(
        input=logps,
        dim=-1,
        index=labels.unsqueeze(1)
    ).squeeze(1)

    if mask is not None:
        mask = mask[:, 1:].clone()
        select_logprobs = select_logprobs * mask
        average_logprobs = select_logprobs.sum(-1) / mask.sum(-1)
        return average_logprobs
    else:
        return select_logprobs.mean(-1)


def compute_tpo_logprobs(logits, labels, mask=None):
    logps = F.log_softmax(logits, dim=-1)
    select_logprobs = torch.gather(
        input=logps,
        dim=-1,
        index=labels.unsqueeze(1)
    ).squeeze(1)

    return select_logprobs


def compute_dpo_logprobs_f_cross(logits, labels, mask=None):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :].clone()

    batch_size, sequence_len, vocab_size = logits.shape
    cross_entropy_loss = 0

    if mask is not None:
        mask = mask[:, 1:].clone()
        labels.masked_fill_(~mask, -100)
        for i in range(batch_size):
            cross_entropy_loss += F.cross_entropy(logits[i], labels[i])
    else:
        for i in range(batch_size):
            cross_entropy_loss += F.cross_entropy(logits[i], labels[i])
    cross_entropy_loss /= batch_size
    return cross_entropy_loss


def compute_dpo_batch_loss(batch, policy_model, reference_model, beta):
    # loss_fn = SimPo(beta, 0.5)   SimPO loss
    loss_fn = DPOLoss(beta)   # DPO loss

    policy_chosen_logps = compute_dpo_logprobs(
        logits=policy_model(batch["chosen"]).logits,
        labels=batch["chosen"],
        mask=batch["chosen_mask"]
    )
    policy_rejected_logps = compute_dpo_logprobs(
        logits=policy_model(batch["rejected"]).logits,
        labels=batch["rejected"],
        mask=batch["rejected_mask"]
    )
    reference_chosen_logps = compute_dpo_logprobs(
        logits=reference_model(batch['chosen'].to(reference_model.device)).logits.to(policy_model.device),
        labels=batch['chosen'],
        mask=batch["chosen_mask"]
    )
    reference_rejected_logps = compute_dpo_logprobs(
        logits=reference_model(batch['rejected'].to(reference_model.device)).logits.to(policy_model.device),
        labels=batch['rejected'],
        mask=batch["rejected_mask"]
    )
    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    # SimPO
    # loss = loss_fn(
    #     policy_chosen_logps=policy_chosen_logps,
    #     policy_rejected_logps=policy_rejected_logps,
    # )
    # return loss
    return loss, chosen_rewards, rejected_rewards


def compute_tpo_batch_loss(batch, policy_model, reference_model, beta):
    loss_fn = TPOLoss(beta)

    batch_tokens = batch['response_list']
    batch_size, n_responses, n_tokens = batch_tokens.shape
    batch_tokens = batch_tokens.view(batch_size * n_responses, n_tokens)
    

    batch_tokens_masks = batch['response_list_mask']
    batch_tokens_masks = batch_tokens_masks.view(batch_size * n_responses, n_tokens)

    policy_out = policy_model(batch_tokens)
    reference_output = reference_model(batch_tokens.to(reference_model.device))

    policy_response_logps = compute_tpo_logprobs(
        logits=policy_out.logits,
        labels=batch_tokens,
        mask=batch_tokens_masks
    )
    reference_response_logps = compute_tpo_logprobs(
        logits=reference_output.logits.to(policy_model.device),
        labels=batch_tokens,
        mask=batch_tokens_masks
    )

    response_embedding = policy_out.hidden_states[-1]

    policy_response_logps = policy_response_logps.view(batch_size, n_responses, n_tokens, 1)
    reference_response_logps = reference_response_logps.view(batch_size, n_responses, n_tokens, 1)
    hidden_state_size = response_embedding.shape[-1]
    response_embedding = response_embedding.view(batch_size, n_responses, n_tokens, hidden_state_size)

    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_responses_logps=policy_response_logps,
        reference_responses_logps=reference_response_logps,
        hidden_state=response_embedding,
        step_index=batch['step_index'],
        labels=batch['score_list']
    )
    return loss, chosen_rewards, rejected_rewards


def compute_loss_dataloader(data_loader, policy_model, reference_model, beta, num_batches=5, method='DPO'):
    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            if method == 'TPO':
                loss, chosen_rewards, rejected_rewards = compute_tpo_batch_loss(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    beta=beta
                )
            else:
                loss, chosen_rewards, rejected_rewards = compute_dpo_batch_loss(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    beta=beta
                )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards
