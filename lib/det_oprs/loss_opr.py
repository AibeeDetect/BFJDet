import torch
import torch.nn as nn
from config_bfj import config

def softmax_loss(score, label, ignore_label=-1, num_classes=2):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def focal_loss(inputs, targets, alpha=-1, gamma=2):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels, num_classes=3)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def embedding_loss(tag0, tag1, cen0, cen1, topk):
    topk2 = topk * topk
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    num = len(tag0)

    # pull body_face
    tag = tag0.unsqueeze(1) - tag1.unsqueeze(2)
    tag = torch.pow(tag, 2) / (num * topk2 + 1e-4)
    pull_bf = tag.sum()

    # pull body_body
    tag_bb = tag0.unsqueeze(1) - tag0.unsqueeze(2)
    cen_bb = cen0[:, :, :2].unsqueeze(1) - cen0[:, :, :2].unsqueeze(2)
    tag_bb = torch.pow(tag_bb, 2) / (num * topk2 + 1e-4)
    cen_bb = torch.sqrt(torch.pow(cen_bb, 2).sum(-1)) / (cen0[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_bb = torch.exp(cen_bb)
    tag_bb = tag_bb * cen_bb.unsqueeze(-1)
    pull_bb = tag_bb.sum()

    # pull face_face
    tag_ff = tag1.unsqueeze(1) - tag1.unsqueeze(2)
    cen_ff = cen1[:, :, :2].unsqueeze(1) - cen1[:, :, :2].unsqueeze(2)
    tag_ff = torch.pow(tag_ff, 2) / (num * topk2 + 1e-4)
    cen_ff = torch.sqrt(torch.pow(cen_ff, 2).sum(-1)) / (cen1[:, :, 2].unsqueeze(-1) + 1e-7)
    cen_ff = torch.exp(cen_ff)
    tag_ff = tag_ff * cen_ff.unsqueeze(-1)
    pull_ff = tag_ff.sum()

    pull = pull_bb * 1.5 + pull_bf + pull_ff * 1.5

    dist1 = tag0.view(-1, tag0.shape[-1])
    dist2 = tag1.view(-1, tag1.shape[-1])
    # push body_face
    push_bf = dist1.unsqueeze(0) - dist2.unsqueeze(1)
    push_bf = torch.pow(push_bf, 2).sum(-1)
    for i in range(int(len(push_bf) / topk)):
        push_bf[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_bf = 2 - push_bf
    push_bf = nn.functional.relu(push_bf, inplace=True)
    push_bf = push_bf / ((num - 1) * num * topk2 + 1e-4)
    push_bf = push_bf.sum()

    # push body_body
    push_bb = dist1.unsqueeze(0) - dist1.unsqueeze(1)
    push_bb = torch.pow(push_bb, 2).sum(-1)
    for i in range(int(len(push_bb) / topk)):
        push_bb[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_bb = 2 - push_bb
    push_bb = nn.functional.relu(push_bb, inplace=True)
    push_bb = push_bb / ((num - 1) * num * topk2 + 1e-4)
    push_bb = push_bb.sum()

    # push face_face
    push_ff = dist2.unsqueeze(0) - dist2.unsqueeze(1)
    push_ff = torch.pow(push_ff, 2).sum(-1)
    for i in range(int(len(push_ff) / topk)):
        push_ff[i*topk:i*topk+topk, i*topk:i*topk+topk] = torch.tensor(0.0).to(device=dist1.device)
    push_ff = 2 - push_ff
    push_ff = nn.functional.relu(push_ff, inplace=True)
    push_ff = push_ff / ((num - 1) * num * topk2 + 1e-4)
    push_ff = push_ff.sum()

    push = push_bb * 1.5 + push_bf + push_ff * 1.5
    # push = push_bf

    return pull, push

# TODO maybe push this to nn?
def angular_loss(input, target):
    input_length = torch.sqrt((input[: ,1] ** 2 + input[:, 0] ** 2)) + 1e-7
    input = input / input_length.unsqueeze(1)
    target_length = torch.sqrt((target[: ,1] ** 2 + target[:, 0] ** 2)) + 1e-7
    target = target / target_length.unsqueeze(1)
    Cross_product = input[:, 1] * target[:, 0] - input[:, 0] * target[:, 1]
    return torch.abs(Cross_product).sum()

if __name__ == "__main__":
    a = torch.Tensor([[1,2],[2,3]])
    b = torch.Tensor([[2,3],[3,4]])
    print(vector_loss(a,b))