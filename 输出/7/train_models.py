import os
import random
import time

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca


TEMPORAL_DIAGNOSTIC_KEYS = [
    "temporal_rescue",
    "temporal_damage",
    "temporal_net_gain",
    "temporal_proposal_rescue",
    "temporal_proposal_damage",
    "temporal_proposal_net_gain",
    "temporal_disagreement",
    "temporal_pair_residual_abs",
    "natural_pair_feedback_scale",
    "natural_pair_feedback_strength_mean",
    "natural_pair_feedback_strength_std",
    "natural_pair_score_std",
    "natural_pair_evidence_abs",
    "natural_pair_top1_agreement",
    "natural_pair_verb_effect_abs",
    "natural_pair_object_effect_abs",
    "temporal_object_prediction_preservation",
    "temporal_only_verb_change",
    "temporal_only_object_change",
    "temporal_both_change",
    "temporal_expert_k3_mean",
    "temporal_expert_k5_mean",
    "temporal_expert_k7_mean",
    "temporal_expert_k9_mean",
    "natural_alpha_verb",
    "natural_alpha_object",
]


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def unpack_c2c_training_outputs(outputs):
    if len(outputs) == 9:
        return (
            outputs,
            None, None, None, None, None, None, None, None, None,
        )
    if len(outputs) == 18:
        return (outputs[:9],) + tuple(outputs[9:])
    raise RuntimeError(
        "Expected 9 base or 18 temporal outputs, got %d."
        % len(outputs)
    )


def ensure_finite_training_loss(total_loss, components, epoch, batch_index):
    if torch.isfinite(total_loss.detach()).all().item():
        return
    component_values = []
    for name, value in components.items():
        detached = value.detach().float()
        if detached.numel() == 1:
            description = str(detached.item())
        else:
            finite_ratio = torch.isfinite(detached).float().mean().item()
            description = "finite_ratio=%.6f" % finite_ratio
        component_values.append("%s=%s" % (name, description))
    raise FloatingPointError(
        "Non-finite training loss at epoch %d batch %d: %s"
        % (epoch, batch_index, ", ".join(component_values))
    )


def temporal_route_loss(gate_weights, gate_prior, eps=1.0e-8):
    if gate_weights is None or gate_prior is None:
        if gate_weights is not None:
            return gate_weights.new_zeros(())
        if gate_prior is not None:
            return gate_prior.new_zeros(())
        return torch.zeros(())
    mean_usage = gate_weights.float().mean(dim=0)
    mean_usage = mean_usage / mean_usage.sum().clamp_min(eps)
    prior = gate_prior.detach().float()
    prior = prior / prior.sum().clamp_min(eps)
    return (
        mean_usage
        * (
            mean_usage.clamp_min(eps).log()
            - prior.clamp_min(eps).log()
        )
    ).sum()


def natural_component_fusion_loss(
        feedback_scores,
        train_attr_indices,
        train_object_indices,
        target_a,
        target_b,
        target_mix,
        cosine_scale):
    """Supervise the final composition score after component feedback."""
    train_scores = feedback_scores[
        :, train_attr_indices, train_object_indices
    ].float() * float(cosine_scale)
    loss_a = F.cross_entropy(train_scores, target_a)
    if target_b is None:
        return loss_a
    mix = float(target_mix)
    loss_b = F.cross_entropy(train_scores, target_b)
    return mix * loss_a + (1.0 - mix) * loss_b


def natural_component_feedback_loss(
        component_scores,
        num_verbs,
        verb_target_a,
        verb_target_b,
        object_target_a,
        object_target_b,
        target_mix,
        logit_scale=50.0,
        verb_loss_weight=0.70,
        object_loss_weight=0.30):
    """Make holistic natural-text evidence predictive of both components."""
    logit_scale = float(logit_scale)
    if logit_scale <= 0.0:
        raise ValueError("Natural component logit scale must be positive.")
    if component_scores is None or component_scores.ndim != 2:
        raise ValueError("component_scores must be a two-dimensional tensor.")
    if not 0 < int(num_verbs) < component_scores.shape[1]:
        raise ValueError("num_verbs does not split component_scores.")
    if verb_loss_weight < 0.0 or object_loss_weight < 0.0:
        raise ValueError("Component loss weights must be non-negative.")
    weight_sum = float(verb_loss_weight + object_loss_weight)
    if weight_sum <= 0.0:
        raise ValueError("At least one component loss weight is required.")
    verb_loss_weight = float(verb_loss_weight) / weight_sum
    object_loss_weight = float(object_loss_weight) / weight_sum

    verb_scores = component_scores[:, :num_verbs].float() * logit_scale
    object_scores = component_scores[:, num_verbs:].float() * logit_scale

    def statistics(scores, labels):
        labels = labels.long()
        loss = F.cross_entropy(scores, labels)
        positive = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        negative_mask = torch.ones_like(scores, dtype=torch.bool)
        negative_mask.scatter_(1, labels.unsqueeze(1), False)
        hard_negative = scores.masked_fill(
            ~negative_mask,
            float('-inf'),
        ).amax(dim=1)
        margin = (positive - hard_negative).mean()
        accuracy = scores.argmax(dim=1).eq(labels).float().mean()
        return loss, margin, accuracy

    verb_a = statistics(verb_scores, verb_target_a)
    object_a = statistics(object_scores, object_target_a)
    if verb_target_b is None or object_target_b is None:
        verb_stats = verb_a
        object_stats = object_a
    else:
        verb_b = statistics(verb_scores, verb_target_b)
        object_b = statistics(object_scores, object_target_b)
        mix = float(target_mix)
        verb_stats = tuple(
            mix * value_a + (1.0 - mix) * value_b
            for value_a, value_b in zip(verb_a, verb_b)
        )
        object_stats = tuple(
            mix * value_a + (1.0 - mix) * value_b
            for value_a, value_b in zip(object_a, object_b)
        )

    component_loss = (
        verb_loss_weight * verb_stats[0]
        + object_loss_weight * object_stats[0]
    )
    return (
        component_loss,
        verb_stats[0],
        object_stats[0],
        verb_stats[1],
        object_stats[1],
        verb_stats[2],
        object_stats[2],
    )


def _clone_parameter_grads(parameters):
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_increments(parameters, previous_grads):
    increments = []
    for parameter, previous in zip(parameters, previous_grads):
        current = parameter.grad
        if current is None:
            increments.append(None)
        elif previous is None:
            increments.append(current.detach().clone())
        else:
            increments.append(current.detach().clone() - previous)
    return increments


def _restore_parameter_grads(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.clone()


def _gradient_dot(left, right):
    value = None
    for left_grad, right_grad in zip(left, right):
        if left_grad is None or right_grad is None:
            continue
        item = (left_grad.float() * right_grad.float()).sum()
        value = item if value is None else value + item
    return value


def _gradient_norm_sq(gradients):
    value = None
    for gradient in gradients:
        if gradient is None:
            continue
        item = gradient.float().square().sum()
        value = item if value is None else value + item
    return value


def global_anchored_backward(
        global_anchor_loss,
        global_remainder_loss,
        temporal_loss,
        scaler,
        shared_parameters,
        shared_gradient_ratio,
        warmup_scale):
    """Backpropagate temporal gradients without opposing the global anchor."""
    if not shared_parameters or not temporal_loss.requires_grad:
        scaler.scale(
            global_anchor_loss + global_remainder_loss + temporal_loss
        ).backward()
        return {
            "cosine": 0.0,
            "conflict": 0.0,
            "raw_ratio": 0.0,
            "applied_ratio": 0.0,
            "warmup": float(warmup_scale),
        }

    previous_grads = _clone_parameter_grads(shared_parameters)
    scaler.scale(global_anchor_loss).backward(retain_graph=True)
    anchor_grads = _gradient_increments(
        shared_parameters,
        previous_grads,
    )
    scaler.scale(global_remainder_loss).backward(retain_graph=True)
    global_grads = _gradient_increments(
        shared_parameters,
        previous_grads,
    )

    _restore_parameter_grads(shared_parameters, previous_grads)
    scaler.scale(temporal_loss).backward()
    temporal_grads = _gradient_increments(
        shared_parameters,
        previous_grads,
    )

    dot = _gradient_dot(anchor_grads, temporal_grads)
    anchor_norm_sq = _gradient_norm_sq(anchor_grads)
    global_norm_sq = _gradient_norm_sq(global_grads)
    temporal_norm_sq = _gradient_norm_sq(temporal_grads)
    eps = 1.0e-12

    dot_value = 0.0 if dot is None else float(dot.detach().item())
    anchor_norm = (
        0.0
        if anchor_norm_sq is None
        else math.sqrt(max(float(anchor_norm_sq.detach().item()), 0.0))
    )
    global_norm = (
        0.0
        if global_norm_sq is None
        else math.sqrt(max(float(global_norm_sq.detach().item()), 0.0))
    )
    temporal_norm = (
        0.0
        if temporal_norm_sq is None
        else math.sqrt(max(float(temporal_norm_sq.detach().item()), 0.0))
    )
    cosine = dot_value / max(anchor_norm * temporal_norm, eps)
    conflict = dot_value < 0.0 and anchor_norm > 0.0
    projection_coefficient = (
        dot_value / max(anchor_norm ** 2, eps)
        if conflict else 0.0
    )

    protected_temporal = []
    for anchor_grad, temporal_grad in zip(anchor_grads, temporal_grads):
        if temporal_grad is None:
            protected_temporal.append(None)
        elif conflict and anchor_grad is not None:
            protected_temporal.append(
                temporal_grad - projection_coefficient * anchor_grad
            )
        else:
            protected_temporal.append(temporal_grad)

    protected_norm_sq = _gradient_norm_sq(protected_temporal)
    protected_norm = (
        0.0
        if protected_norm_sq is None
        else math.sqrt(max(float(protected_norm_sq.detach().item()), 0.0))
    )
    maximum_temporal_norm = float(shared_gradient_ratio) * global_norm
    norm_scale = min(
        1.0,
        maximum_temporal_norm / max(protected_norm, eps),
    )
    applied_scale = float(warmup_scale) * norm_scale

    combined_grads = []
    for previous, global_grad, temporal_grad in zip(
            previous_grads,
            global_grads,
            protected_temporal):
        combined = None if previous is None else previous.clone()
        if global_grad is not None:
            combined = (
                global_grad.clone()
                if combined is None
                else combined + global_grad
            )
        if temporal_grad is not None and applied_scale > 0.0:
            applied = temporal_grad * applied_scale
            combined = applied if combined is None else combined + applied
        combined_grads.append(combined)
    _restore_parameter_grads(shared_parameters, combined_grads)

    applied_temporal_norm = protected_norm * applied_scale
    return {
        "cosine": cosine,
        "conflict": float(conflict),
        "raw_ratio": temporal_norm / max(global_norm, eps),
        "applied_ratio": applied_temporal_norm / max(global_norm, eps),
        "warmup": float(warmup_scale),
    }


def temporal_change_statistics(
        global_scores,
        changed_scores,
        target_indices):
    global_correct = global_scores.detach().argmax(dim=1).eq(target_indices)
    changed_correct = changed_scores.detach().argmax(dim=1).eq(target_indices)
    rescue = (changed_correct & ~global_correct).float().mean()
    damage = (global_correct & ~changed_correct).float().mean()
    disagreement = (
        global_scores.detach().argmax(dim=1)
        != changed_scores.detach().argmax(dim=1)
    ).float().mean()
    return rescue, damage, disagreement


def evaluate(model, dataset, config, fusion_alphas=None):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    branch_diagnostics_enabled = bool(getattr(
        config, 'temporal_composition_enabled', False
    ))
    prediction_outputs = test.predict_logits(
        model,
        dataset,
        config,
        return_branch_diagnostics=branch_diagnostics_enabled,
    )
    if len(prediction_outputs) == 6:
        (
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            loss_avg,
            branch_diagnostics,
        ) = prediction_outputs
    else:
        (
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            loss_avg,
        ) = prediction_outputs
        branch_diagnostics = {}

    # Drop internal tensors kept only for optional post-hoc analysis.
    branch_diagnostics.pop("_global_logits", None)
    branch_diagnostics.pop("_verb_evidence_logits", None)
    branch_diagnostics.pop("_object_evidence_logits", None)

    # Inference uses the model's fixed (verb, object) fusion strengths.
    # fusion_alphas is accepted only for API compatibility / logging.
    core_model = model.module if hasattr(model, 'module') else model
    fusion = getattr(
        core_model,
        'c2c_natural_pair_probabilistic_fusion',
        None,
    )
    if fusion_alphas is not None:
        alpha_verb, alpha_object = fusion_alphas
    elif fusion is not None:
        alpha_verb = float(fusion.verb_feedback_strength)
        alpha_object = float(fusion.object_feedback_strength)
    else:
        alpha_verb = float(getattr(
            config, 'natural_pair_verb_feedback_strength', 0.0
        ))
        alpha_object = float(getattr(
            config, 'natural_pair_object_feedback_strength', 0.0
        ))

    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config,
    )
    test_stats["natural_alpha_verb"] = float(alpha_verb)
    test_stats["natural_alpha_object"] = float(alpha_object)
    test_stats.update(branch_diagnostics)
    result = ""
    key_set = [
        "attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all",
        "best_seen", "best_unseen", "best_hm", "AUC",
    ]

    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    if branch_diagnostics:
        diagnostic_result = " | ".join(
            f"{key} {branch_diagnostics[key]:.6f}"
            for key in TEMPORAL_DIAGNOSTIC_KEYS
            if key in branch_diagnostics
        )
        print(diagnostic_result)
    model.train()
    return loss_avg, test_stats


def parse_fusion_alpha_grid(config):
    """Deprecated: fixed inference alphas replaced the validation grid."""
    return None

def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


# ========conditional train=
def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        epoch_train_start = time.perf_counter()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()
            with torch.cuda.amp.autocast(enabled=True):
                model_outputs = model(batch_img)
                base_outputs, *_ = (
                    unpack_c2c_training_outputs(model_outputs)
                )
                p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = base_outputs
                # component loss
                loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)
                loss = loss_com + 0.2 * (loss_verb + loss_obj)

                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_seconds = time.perf_counter() - epoch_train_start
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(
            f"epoch {i + 1} train seconds {epoch_train_seconds:.2f}\n"
        )
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
        key_set = [
            "attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all",
            "best_seen", "best_unseen", "best_hm", "AUC",
        ] + TEMPORAL_DIAGNOSTIC_KEYS
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        key_set = [
            "attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all",
            "best_seen", "best_unseen", "best_hm", "AUC",
        ] + TEMPORAL_DIAGNOSTIC_KEYS
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))


def c2c_enhance(
        model,
        optimizer,
        lr_scheduler,
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        scaler):
    """Train global C2C with calibrated natural joint-text evidence."""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    model.train()
    best_loss = 1e5
    best_metric = 0
    loss_fn = CrossEntropyLoss()
    log_training = open(
        os.path.join(config.save_path, 'log.txt'),
        'w',
    )

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    v_o_on_v, v_o_on_o = cal_conditional(
        attr2idx,
        obj2idx,
        'train',
        train_dataset,
    )
    v_o_on_v = v_o_on_v.cuda()
    v_o_on_o = v_o_on_o.cuda()
    train_pairs = torch.tensor(
        [
            (attr2idx[attr], obj2idx[obj])
            for attr, obj in train_dataset.train_pairs
        ],
        dtype=torch.long,
        device='cuda',
    )
    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]

    natural_feedback_weight = float(getattr(
        config,
        'natural_pair_feedback_loss_weight',
        0.0,
    ))
    natural_component_weight = float(getattr(
        config,
        'natural_pair_component_loss_weight',
        0.25,
    ))
    natural_component_logit_scale = float(getattr(
        config,
        'natural_pair_component_logit_scale',
        50.0,
    ))
    natural_verb_component_weight = float(getattr(
        config,
        'natural_pair_verb_component_loss_weight',
        0.70,
    ))
    natural_object_component_weight = float(getattr(
        config,
        'natural_pair_object_component_loss_weight',
        0.30,
    ))
    temporal_route_weight = float(getattr(
        config,
        'temporal_composition_route_loss_weight',
        0.02,
    ))
    temporal_shaper_weight = float(getattr(
        config,
        'temporal_shaper_loss_weight',
        0.50,
    ))
    temporal_gradient_clip = float(getattr(
        config,
        'temporal_gradient_clip',
        5.0,
    ))
    if min(
            natural_feedback_weight,
            natural_component_weight,
            natural_verb_component_weight,
            natural_object_component_weight,
            temporal_route_weight,
            temporal_shaper_weight) < 0.0:
        raise ValueError("Temporal loss weights must be non-negative.")
    if natural_component_logit_scale <= 0.0:
        raise ValueError(
            "natural_pair_component_logit_scale must be positive."
        )
    if (
            natural_verb_component_weight
            + natural_object_component_weight
    ) <= 0.0:
        raise ValueError(
            "At least one natural component loss weight must be positive."
        )

    core_model = model.module if hasattr(model, 'module') else model
    temporal_parameter_markers = (
        "c2c_temporal_composition_expert",
        "c2c_natural_pair_probabilistic_fusion",
        "c2c_temporal_relation_shaper",
        "c2c_temporal_shaper_projection",
    )
    temporal_branch_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(marker in name for marker in temporal_parameter_markers)
    ]
    temporal_enabled = bool(getattr(
        config,
        'temporal_composition_enabled',
        False,
    ))
    if temporal_enabled and not temporal_branch_parameters:
        raise RuntimeError(
            "Temporal composition is enabled but its relation parameters "
            "were not found."
        )
    gate_prior = None
    if temporal_enabled:
        core_model.prepare_natural_pair_text_bank()
        gate_prior = (
            core_model.c2c_temporal_composition_expert.gate_prior
        )

    key_set = [
        "attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all",
        "best_seen", "best_unseen", "best_hm", "AUC",
    ] + TEMPORAL_DIAGNOSTIC_KEYS

    def write_result(result_values, prefix):
        result_line = ""
        for key, value in result_values.items():
            if key in key_set:
                result_line += key + "  " + str(round(value, 4)) + "| "
        log_training.write('\n' + result_line + '\n')
        print("{} Loss average: {}".format(prefix, current_loss_avg))
        log_training.write(
            "{} Loss average: {}\n".format(prefix, current_loss_avg)
        )

    optimizer.zero_grad()
    for epoch_index in range(config.epoch_start, config.epochs):
        epoch_start_time = time.perf_counter()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader),
            desc="epoch % 3d" % (epoch_index + 1),
        )
        scalar_stats = {
            "train": [],
            "composition": [],
            "verb": [],
            "object": [],
            "hsic_v": [],
            "hsic_o": [],
            "hsic_vo": [],
            "condition": [],
            "natural_feedback": [],
            "natural_component": [],
            "temporal_shaper": [],
            "component_verb": [],
            "component_object": [],
            "component_verb_margin": [],
            "component_object_margin": [],
            "component_verb_accuracy": [],
            "component_object_accuracy": [],
            "temporal_route": [],
            "natural_score_std": [],
            "natural_evidence_abs": [],
            "natural_top1_agreement": [],
            "natural_feedback_scale": [],
            "natural_feedback_strength": [],
            "natural_verb_evidence_abs": [],
            "natural_object_evidence_abs": [],
            "residual_abs": [],
            "rescue": [],
            "damage": [],
            "disagreement": [],
            "proposal_rescue": [],
            "proposal_damage": [],
        }
        gate_sum = None
        gate_min = None
        gate_max = None
        gate_entropy_sum = None
        gate_maximum_sum = None
        gate_sample_count = 0
        nonfinite_temporal_gradient_steps = 0

        print("Current_lr:{}".format(optimizer.param_groups[-1]['lr']))
        for batch_index, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            condition_weight = float(getattr(
                config,
                'condition_loss_weight',
                0.05,
            ))

            with torch.cuda.amp.autocast(enabled=True):
                use_cutmix = np.random.rand(1) < config.cutmix_prob
                if use_cutmix:
                    mix = float(np.random.beta(config.beta, config.beta))
                    rand_index = torch.randperm(
                        batch_verb.shape[0],
                        device=batch_verb.device,
                    )
                    target_o_a = batch_obj
                    target_o_b = batch_obj[rand_index]
                    target_v_a = batch_verb
                    target_v_b = batch_verb[rand_index]
                    target_pair_a = batch_target
                    target_pair_b = batch_target[rand_index]
                    target_all_a = target_v_a * len(obj2idx) + target_o_b
                    target_all_b = target_v_b * len(obj2idx) + target_o_a

                    bbx1, bby1, bbx2, bby2 = rand_bbox(
                        batch_img.size(),
                        mix,
                    )
                    batch_img[
                        :, :, :, bbx1:bbx2, bby1:bby2
                    ] = batch_img[
                        rand_index, :, :, bbx1:bbx2, bby1:bby2
                    ]
                    mix = 1.0 - (
                        (bbx2 - bbx1) * (bby2 - bby1)
                        / (
                            batch_img.size()[-1]
                            * batch_img.size()[-2]
                        )
                    )
                else:
                    mix = 1.0
                    target_o_a = batch_obj
                    target_o_b = None
                    target_v_a = batch_verb
                    target_v_b = None
                    target_pair_a = batch_target
                    target_pair_b = None
                    target_all_a = None
                    target_all_b = None

                model_outputs = model(batch_img)
                (
                    base_outputs,
                    global_pair_scores,
                    temporal_proposal_scores,
                    temporal_corrected_scores,
                    temporal_gate_weights,
                    temporal_relation_diagnostics,
                    temporal_feedback_scale,
                    temporal_feedback_strength_value,
                    temporal_component_feedback_scores,
                    temporal_shaper_scores,
                ) = unpack_c2c_training_outputs(model_outputs)
                (
                    p_v,
                    p_o,
                    p_pair_v,
                    p_pair_o,
                    vid_feat,
                    v_feat,
                    o_feat,
                    p_v_con_o,
                    p_o_con_v,
                ) = base_outputs

                if use_cutmix:
                    loss_verb = (
                        loss_fn(
                            p_v * config.cosine_scale,
                            target_v_a,
                        ) * mix
                        + loss_fn(
                            p_v * config.cosine_scale,
                            target_v_b,
                        ) * (1.0 - mix)
                    )
                    loss_obj = (
                        loss_fn(
                            p_o * config.cosine_scale,
                            target_o_a,
                        ) * mix
                        + loss_fn(
                            p_o * config.cosine_scale,
                            target_o_b,
                        ) * (1.0 - mix)
                    )
                    pred_com_train = (
                        p_pair_v[:, train_v_inds, train_o_inds]
                        + p_pair_o[:, train_v_inds, train_o_inds]
                    )
                    loss_com_train = (
                        loss_fn(
                            pred_com_train * config.cosine_scale,
                            target_pair_a,
                        ) * mix
                        + loss_fn(
                            pred_com_train * config.cosine_scale,
                            target_pair_b,
                        ) * (1.0 - mix)
                    )
                    pred_com_all = (p_pair_v + p_pair_o).reshape(
                        batch_verb.shape[0],
                        -1,
                    )
                    loss_com_all = (
                        loss_fn(
                            pred_com_all * config.cosine_scale,
                            target_all_a,
                        )
                        + loss_fn(
                            pred_com_all * config.cosine_scale,
                            target_all_b,
                        )
                    )
                    loss_com = (
                        loss_com_train
                        + condition_weight * loss_com_all
                    )
                else:
                    loss_verb = loss_fn(
                        p_v * config.cosine_scale,
                        target_v_a,
                    )
                    loss_obj = loss_fn(
                        p_o * config.cosine_scale,
                        target_o_a,
                    )
                    pred_com_train = (
                        p_pair_v[:, train_v_inds, train_o_inds]
                        + p_pair_o[:, train_v_inds, train_o_inds]
                    )
                    loss_com = loss_fn(
                        pred_com_train * config.cosine_scale,
                        target_pair_a,
                    )

                if use_cutmix:
                    object_targets = (
                        mix
                        * F.one_hot(
                            target_o_a,
                            len(obj2idx),
                        ).float()
                        + (1.0 - mix)
                        * F.one_hot(
                            target_o_b,
                            len(obj2idx),
                        ).float()
                    )
                    verb_targets = (
                        mix
                        * F.one_hot(
                            target_v_a,
                            len(attr2idx),
                        ).float()
                        + (1.0 - mix)
                        * F.one_hot(
                            target_v_b,
                            len(attr2idx),
                        ).float()
                    )
                else:
                    object_targets = F.one_hot(
                        target_o_a,
                        len(obj2idx),
                    ).float()
                    verb_targets = F.one_hot(
                        target_v_a,
                        len(attr2idx),
                    ).float()

                pooled_video = vid_feat.mean(-1)
                loss_hsic_v = (
                    hsic_normalized_cca(pooled_video, v_feat, 20)
                    - hsic_normalized_cca(v_feat, verb_targets, 20)
                )
                loss_hsic_o = (
                    hsic_normalized_cca(pooled_video, o_feat, 20)
                    - hsic_normalized_cca(o_feat, object_targets, 20)
                )
                half_dim = int(v_feat.shape[-1] * 0.5)
                loss_hsic_vo = hsic_normalized_cca(
                    v_feat[:, :half_dim],
                    o_feat[:, :half_dim],
                    20,
                )
                loss_hsic = (
                    loss_hsic_v + loss_hsic_o + loss_hsic_vo
                )

                if use_cutmix:
                    loss_condition = loss_com.new_zeros(())
                else:
                    loss_on_v = loss_fn(
                        p_o_con_v.mean(0),
                        v_o_on_v,
                    )
                    loss_on_o = loss_fn(
                        p_v_con_o.mean(0).permute(1, 0),
                        v_o_on_o.permute(1, 0),
                    )
                    loss_condition = loss_on_o + loss_on_v

                zero = loss_com.new_zeros(())
                loss_natural_feedback = zero
                loss_natural_component = zero
                loss_temporal_shaper = zero
                loss_natural_component_verb = zero
                loss_natural_component_object = zero
                component_verb_margin = zero
                component_object_margin = zero
                component_verb_accuracy = zero
                component_object_accuracy = zero
                loss_temporal_route = zero
                final_rescue = zero
                final_damage = zero
                final_disagreement = zero
                proposal_rescue = zero
                proposal_damage = zero
                residual_abs = zero
                natural_score_std = zero
                natural_evidence_abs = zero
                natural_top1_agreement = zero
                natural_feedback_scale_mean = zero
                natural_feedback_strength_mean = zero
                natural_verb_evidence_abs = zero
                natural_object_evidence_abs = zero
                if temporal_corrected_scores is not None:
                    if temporal_component_feedback_scores is None:
                        raise RuntimeError(
                            "Natural temporal branch did not return its "
                            "component feedback scores."
                        )
                    loss_natural_feedback = zero
                    if natural_feedback_weight > 0.0:
                        loss_natural_feedback = natural_component_fusion_loss(
                            temporal_corrected_scores,
                            train_v_inds,
                            train_o_inds,
                            target_pair_a,
                            target_pair_b,
                            mix,
                            config.cosine_scale,
                        )
                    (
                        loss_natural_component,
                        loss_natural_component_verb,
                        loss_natural_component_object,
                        component_verb_margin,
                        component_object_margin,
                        component_verb_accuracy,
                        component_object_accuracy,
                    ) = natural_component_feedback_loss(
                        temporal_component_feedback_scores,
                        len(attr2idx),
                        target_v_a,
                        target_v_b,
                        target_o_a,
                        target_o_b,
                        mix,
                        logit_scale=natural_component_logit_scale,
                        verb_loss_weight=natural_verb_component_weight,
                        object_loss_weight=natural_object_component_weight,
                    )
                    loss_temporal_route = temporal_route_loss(
                        temporal_gate_weights,
                        gate_prior,
                    ).to(device=loss_com.device)
                    if temporal_shaper_scores is not None:
                        shaper_train_scores = (
                            temporal_shaper_scores.float()
                            * config.cosine_scale
                        )
                        loss_temporal_shaper = F.cross_entropy(
                            shaper_train_scores,
                            target_pair_a,
                        )
                        if target_pair_b is not None:
                            loss_temporal_shaper = (
                                mix * loss_temporal_shaper
                                + (1.0 - mix) * F.cross_entropy(
                                    shaper_train_scores,
                                    target_pair_b,
                                )
                            )

                    global_train_scores = global_pair_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    corrected_train_scores = temporal_corrected_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    proposal_train_scores = temporal_proposal_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    # When training without fusion, corrected == global, so
                    # report the counterfactual fixed-alpha proposal as the
                    # primary temporal correction diagnostic.
                    diagnostic_train_scores = (
                        proposal_train_scores
                        if not bool(getattr(
                            core_model,
                            'natural_pair_fuse_during_training',
                            False,
                        ))
                        else corrected_train_scores
                    )
                    final_a = temporal_change_statistics(
                        global_train_scores,
                        diagnostic_train_scores,
                        target_pair_a,
                    )
                    proposal_a = temporal_change_statistics(
                        global_train_scores,
                        proposal_train_scores,
                        target_pair_a,
                    )
                    if target_pair_b is None:
                        final_b = final_a
                        proposal_b = proposal_a
                    else:
                        final_b = temporal_change_statistics(
                            global_train_scores,
                            diagnostic_train_scores,
                            target_pair_b,
                        )
                        proposal_b = temporal_change_statistics(
                            global_train_scores,
                            proposal_train_scores,
                            target_pair_b,
                        )
                    (
                        final_rescue,
                        final_damage,
                        final_disagreement,
                    ) = tuple(
                        mix * value_a + (1.0 - mix) * value_b
                        for value_a, value_b in zip(final_a, final_b)
                    )
                    (
                        proposal_rescue,
                        proposal_damage,
                        _proposal_disagreement,
                    ) = tuple(
                        mix * value_a + (1.0 - mix) * value_b
                        for value_a, value_b in zip(
                            proposal_a,
                            proposal_b,
                        )
                    )
                    residual_abs = (
                        temporal_proposal_scores.float()
                        - global_pair_scores.detach().float()
                    ).abs().mean()
                    relation_stats = (
                        temporal_relation_diagnostics.float().mean(dim=0)
                    )
                    natural_score_std = relation_stats[0]
                    natural_evidence_abs = relation_stats[1]
                    natural_top1_agreement = relation_stats[2]
                    natural_feedback_strength_mean = relation_stats[3]
                    natural_verb_evidence_abs = relation_stats[4]
                    natural_object_evidence_abs = relation_stats[5]
                    natural_feedback_scale_mean = (
                        temporal_feedback_scale.float().mean()
                    )

                loss_global = (
                    loss_com
                    + 0.2 * loss_verb
                    + 0.2 * loss_obj
                    + 0.1 * loss_hsic
                    + condition_weight * loss_condition
                )
                loss_temporal = (
                    natural_component_weight * loss_natural_component
                    + natural_feedback_weight * loss_natural_feedback
                    + temporal_route_weight * loss_temporal_route
                    + temporal_shaper_weight * loss_temporal_shaper
                )
                unscaled_loss = loss_global + loss_temporal
                ensure_finite_training_loss(
                    unscaled_loss,
                    {
                        "composition": loss_com,
                        "verb": loss_verb,
                        "object": loss_obj,
                        "hsic": loss_hsic,
                        "condition": loss_condition,
                        "natural_feedback": loss_natural_feedback,
                        "natural_component": loss_natural_component,
                        "temporal_route": loss_temporal_route,
                        "temporal_shaper": loss_temporal_shaper,
                    },
                    epoch_index + 1,
                    batch_index + 1,
                )
                loss = (
                    unscaled_loss
                    / float(config.gradient_accumulation_steps)
                )

            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_dataloader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                temporal_gradients_finite = all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all().item()
                    for parameter in temporal_branch_parameters
                )
                if temporal_branch_parameters and temporal_gradients_finite:
                    torch.nn.utils.clip_grad_norm_(
                        temporal_branch_parameters,
                        max_norm=temporal_gradient_clip,
                        error_if_nonfinite=True,
                    )
                elif temporal_branch_parameters:
                    # GradScaler has already recorded the overflow in
                    # unscale_ and will skip this optimizer update safely.
                    nonfinite_temporal_gradient_steps += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scalar_stats["train"].append(loss.item())
            scalar_stats["composition"].append(loss_com.item())
            scalar_stats["verb"].append(loss_verb.item())
            scalar_stats["object"].append(loss_obj.item())
            scalar_stats["hsic_v"].append(loss_hsic_v.item())
            scalar_stats["hsic_o"].append(loss_hsic_o.item())
            scalar_stats["hsic_vo"].append(loss_hsic_vo.item())
            scalar_stats["condition"].append(loss_condition.item())
            scalar_stats["natural_feedback"].append(
                loss_natural_feedback.detach().float().item()
            )
            scalar_stats["natural_component"].append(
                loss_natural_component.detach().float().item()
            )
            scalar_stats["component_verb"].append(
                loss_natural_component_verb.detach().float().item()
            )
            scalar_stats["component_object"].append(
                loss_natural_component_object.detach().float().item()
            )
            scalar_stats["component_verb_margin"].append(
                component_verb_margin.detach().float().item()
            )
            scalar_stats["component_object_margin"].append(
                component_object_margin.detach().float().item()
            )
            scalar_stats["component_verb_accuracy"].append(
                component_verb_accuracy.detach().float().item()
            )
            scalar_stats["component_object_accuracy"].append(
                component_object_accuracy.detach().float().item()
            )
            scalar_stats["temporal_route"].append(
                loss_temporal_route.detach().float().item()
            )
            scalar_stats["temporal_shaper"].append(
                loss_temporal_shaper.detach().float().item()
            )
            scalar_stats["natural_score_std"].append(
                natural_score_std.detach().float().item()
            )
            scalar_stats["natural_evidence_abs"].append(
                natural_evidence_abs.detach().float().item()
            )
            scalar_stats["natural_top1_agreement"].append(
                natural_top1_agreement.detach().float().item()
            )
            scalar_stats["natural_feedback_scale"].append(
                natural_feedback_scale_mean.detach().float().item()
            )
            scalar_stats["natural_feedback_strength"].append(
                natural_feedback_strength_mean.detach().float().item()
            )
            scalar_stats["natural_verb_evidence_abs"].append(
                natural_verb_evidence_abs.detach().float().item()
            )
            scalar_stats["natural_object_evidence_abs"].append(
                natural_object_evidence_abs.detach().float().item()
            )
            scalar_stats["residual_abs"].append(
                residual_abs.detach().float().item()
            )
            scalar_stats["rescue"].append(
                final_rescue.detach().float().item()
            )
            scalar_stats["damage"].append(
                final_damage.detach().float().item()
            )
            scalar_stats["disagreement"].append(
                final_disagreement.detach().float().item()
            )
            scalar_stats["proposal_rescue"].append(
                proposal_rescue.detach().float().item()
            )
            scalar_stats["proposal_damage"].append(
                proposal_damage.detach().float().item()
            )

            if temporal_gate_weights is not None:
                gate_values = temporal_gate_weights.detach().float()
                batch_gate_sum = gate_values.sum(dim=0)
                batch_gate_min = gate_values.amin(dim=0)
                batch_gate_max = gate_values.amax(dim=0)
                batch_entropy = -(
                    gate_values
                    * gate_values.clamp_min(1.0e-8).log()
                ).sum()
                batch_maximum = gate_values.max(dim=1).values.sum()
                if gate_sum is None:
                    gate_sum = batch_gate_sum
                    gate_min = batch_gate_min
                    gate_max = batch_gate_max
                    gate_entropy_sum = batch_entropy
                    gate_maximum_sum = batch_maximum
                else:
                    gate_sum += batch_gate_sum
                    gate_min = torch.minimum(gate_min, batch_gate_min)
                    gate_max = torch.maximum(gate_max, batch_gate_max)
                    gate_entropy_sum += batch_entropy
                    gate_maximum_sum += batch_maximum
                gate_sample_count += gate_values.shape[0]

            progress_bar.set_postfix({
                "train loss": np.mean(scalar_stats["train"][-50:])
            })
            progress_bar.update()

        lr_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start_time
        progress_bar.close()

        def stat_mean(name):
            values = scalar_stats[name]
            return float(np.mean(values)) if values else 0.0

        train_message = (
            "epoch {} train loss {}".format(
                epoch_index + 1,
                stat_mean("train"),
            )
        )
        progress_bar.write(train_message)
        log_training.write('\n' + train_message + '\n')
        log_training.write(
            "epoch {} train seconds {:.2f}\n".format(
                epoch_index + 1,
                epoch_seconds,
            )
        )
        log_training.write(
            "epoch {} AMP nonfinite temporal gradient steps {}\n".format(
                epoch_index + 1,
                nonfinite_temporal_gradient_steps,
            )
        )
        for label, stat_name in (
                ("com", "composition"),
                ("vv", "verb"),
                ("oo", "object"),
                ("hsic_v", "hsic_v"),
                ("hsic_o", "hsic_o"),
                ("hsic_vo", "hsic_vo"),
                ("con_train", "condition")):
            log_training.write(
                "epoch {} {} loss {}\n".format(
                    epoch_index + 1,
                    label,
                    stat_mean(stat_name),
                )
            )
        log_training.write(
            "epoch {} natural feedback/component/route/shaper loss "
            "{:.6f}/{:.6f}/{:.6f}/{:.6f}\n".format(
                epoch_index + 1,
                stat_mean("natural_feedback"),
                stat_mean("natural_component"),
                stat_mean("temporal_route"),
                stat_mean("temporal_shaper"),
            )
        )
        log_training.write(
            "epoch {} natural component verb/object loss {:.6f}/{:.6f} | "
            "hard margin {:.6f}/{:.6f} | component top1 {:.6f}/{:.6f}\n"
            .format(
                epoch_index + 1,
                stat_mean("component_verb"),
                stat_mean("component_object"),
                stat_mean("component_verb_margin"),
                stat_mean("component_object_margin"),
                stat_mean("component_verb_accuracy"),
                stat_mean("component_object_accuracy"),
            )
        )
        log_training.write(
            "epoch {} natural pair score std/component evidence abs "
            "{:.6f}/{:.6f} | feedback scale {:.6f} | residual abs {:.6f}\n"
            .format(
                epoch_index + 1,
                stat_mean("natural_score_std"),
                stat_mean("natural_evidence_abs"),
                stat_mean("natural_feedback_scale"),
                stat_mean("residual_abs"),
            )
        )
        log_training.write(
            "epoch {} natural pair top1 agreement {:.6f} | "
            "fixed feedback strength {:.6f} | raw verb/object evidence "
            "{:.6f}/{:.6f}\n".format(
                epoch_index + 1,
                stat_mean("natural_top1_agreement"),
                stat_mean("natural_feedback_strength"),
                stat_mean("natural_verb_evidence_abs"),
                stat_mean("natural_object_evidence_abs"),
            )
        )
        log_training.write(
            "epoch {} temporal rescue/damage/net "
            "{:.6f}/{:.6f}/{:.6f} | disagreement {:.6f}\n".format(
                epoch_index + 1,
                stat_mean("rescue"),
                stat_mean("damage"),
                stat_mean("rescue") - stat_mean("damage"),
                stat_mean("disagreement"),
            )
        )
        log_training.write(
            "epoch {} temporal proposal rescue/damage/net "
            "{:.6f}/{:.6f}/{:.6f}\n".format(
                epoch_index + 1,
                stat_mean("proposal_rescue"),
                stat_mean("proposal_damage"),
                stat_mean("proposal_rescue")
                - stat_mean("proposal_damage"),
            )
        )

        if gate_sample_count > 0:
            gate_mean = (gate_sum / float(gate_sample_count)).cpu()
            gate_min_cpu = gate_min.cpu()
            gate_max_cpu = gate_max.cpu()
            gate_entropy = (
                gate_entropy_sum / float(gate_sample_count)
            ).item()
            gate_maximum = (
                gate_maximum_sum / float(gate_sample_count)
            ).item()
            gate_message = " | ".join(
                "k{}:{:.6f}".format(kernel, value)
                for kernel, value in zip(
                    (3, 5, 7, 9),
                    gate_mean.tolist(),
                )
            )
            progress_bar.write(
                "epoch {} temporal gate mean {}".format(
                    epoch_index + 1,
                    gate_message,
                )
            )
            log_training.write(
                "epoch {} temporal gate entropy {:.6f} | max {:.6f}\n".format(
                    epoch_index + 1,
                    gate_entropy,
                    gate_maximum,
                )
            )
            log_training.write(
                "epoch {} temporal gate mean {}\n".format(
                    epoch_index + 1,
                    gate_message,
                )
            )
            gate_range = " | ".join(
                "k{}:{:.6f}/{:.6f}/{:.6f}".format(
                    kernel,
                    minimum,
                    mean,
                    maximum,
                )
                for kernel, minimum, mean, maximum in zip(
                    (3, 5, 7, 9),
                    gate_min_cpu.tolist(),
                    gate_mean.tolist(),
                    gate_max_cpu.tolist(),
                )
            )
            log_training.write(
                "epoch {} temporal gate min/mean/max {}\n".format(
                    epoch_index + 1,
                    gate_range,
                )
            )

        log_training.write(
            "epoch {} temporal branch settings adaptive temperature {:.4f} | "
            "role-robust natural joint text | train-without-fusion {} | "
            "inference verb/object strengths {:.3f}/{:.3f} | "
            "prompt weights {} | decomposition iterations {} | "
            "component/feedback/route/shaper loss weights "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f} | verb/object component mix "
            "{:.3f}/{:.3f} | component logit scale {:.4f} | "
            "feature residual {:.4f} | backbone gradient scale {:.4f} | "
            "relation shaper {}\n"
            .format(
                epoch_index + 1,
                core_model.c2c_temporal_composition_expert.gate_temperature,
                (not core_model.natural_pair_fuse_during_training),
                core_model.natural_pair_verb_feedback_strength,
                core_model.natural_pair_object_feedback_strength,
                ",".join(
                    "{:.2f}".format(weight)
                    for weight in core_model.natural_pair_prompt_weights
                ),
                core_model.natural_pair_decomposition_iterations,
                natural_component_weight,
                natural_feedback_weight,
                temporal_route_weight,
                temporal_shaper_weight,
                natural_verb_component_weight,
                natural_object_component_weight,
                natural_component_logit_scale,
                core_model.temporal_composition_feature_residual,
                core_model.temporal_composition_backbone_gradient_scale,
                core_model.temporal_relation_shaper_enabled,
            )
        )

        if (epoch_index + 1) % config.save_every_n == 0:
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                },
                config.save_path,
                epoch_index,
            )

        should_evaluate = (
            epoch_index % config.eval_every_n == 0
            or epoch_index + 1 == config.epochs
            or epoch_index >= config.val_epochs_ts
        )
        if should_evaluate:
            print("Evaluating val dataset:")
            current_loss_avg, val_result = evaluate(
                model,
                val_dataset,
                config,
            )
            write_result(val_result, "Validation")
            is_best = False
            if config.best_model_metric == "best_loss":
                if current_loss_avg.cpu().float() < best_loss:
                    best_loss = current_loss_avg.cpu().float()
                    is_best = True
            elif val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                is_best = True

            if is_best:
                print("find best!")
                log_training.write("find best!\n")
                current_loss_avg, test_result = evaluate(
                    model,
                    test_dataset,
                    config,
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(config.save_path, "best.pt"),
                )
                write_result(test_result, "Test")

        log_training.write('\n')
        log_training.flush()

        if epoch_index + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path,
                "best.pt",
            )))
            current_loss_avg, test_result = evaluate(
                model,
                test_dataset,
                config,
            )
            write_result(test_result, "Final test")
            log_training.flush()

def c2c_enhance_legacy_dense_residual(
        model,
        optimizer,
        lr_scheduler,
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    from utils.hsic import hsic_normalized_cca
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    v_o_on_v, v_o_on_o = cal_conditional(attr2idx, obj2idx, 'train', train_dataset)
    v_o_on_v, v_o_on_o = v_o_on_v.cuda(), v_o_on_o.cuda()

    # loss = loss_com
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
    temporal_boost_weight = float(getattr(
        config, 'temporal_boost_loss_weight', 0.20
    ))
    temporal_fused_weight = float(getattr(
        config, 'temporal_fused_loss_weight', 0.20
    ))
    temporal_shared_gradient_ratio = float(getattr(
        config, 'temporal_shared_gradient_ratio', 0.25
    ))
    temporal_shared_warmup_epochs = float(getattr(
        config, 'temporal_shared_warmup_epochs', config.warmup
    ))
    temporal_gradient_clip = float(getattr(
        config, 'temporal_transfer_gradient_clip', 5.0
    ))
    if temporal_boost_weight < 0.0 or temporal_fused_weight < 0.0:
        raise ValueError(
            "Temporal residual loss weights must be non-negative."
        )
    if not 0.0 <= temporal_shared_gradient_ratio <= 1.0:
        raise ValueError("temporal_shared_gradient_ratio must lie in [0, 1].")
    if temporal_shared_warmup_epochs < 0.0:
        raise ValueError("temporal_shared_warmup_epochs must be non-negative.")
    if temporal_gradient_clip <= 0.0:
        raise ValueError(
            "temporal_transfer_gradient_clip must be positive."
        )
    temporal_parameter_markers = (
        "c2c_temporal_composition_expert",
        "c2c_global_anchored_temporal_residual",
    )
    temporal_branch_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(marker in name for marker in temporal_parameter_markers)
    ]
    if bool(getattr(config, 'temporal_composition_enabled', False)):
        if not temporal_branch_parameters:
            raise RuntimeError(
                "Temporal composition is enabled but no trainable temporal "
                "branch parameters were found."
            )
        if bool(getattr(config, 'temporal_composition_detach_backbone', False)):
            raise ValueError(
                "Global-anchored residual training requires "
                "temporal_composition_detach_backbone=false."
            )
    shared_video_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "video_encoder" in name
    ]
    if bool(getattr(config, 'temporal_composition_enabled', False)):
        if not shared_video_parameters:
            raise RuntimeError(
                "No trainable shared video parameters were found for "
                "temporal gradient protection."
            )

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        epoch_train_start = time.perf_counter()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []
        epoch_hsic_v_losses = []
        epoch_hsic_o_losses = []
        epoch_hsic_vo_losses = []
        epoch_con_train_losses = []
        epoch_temporal_loss_sum = None
        epoch_temporal_boost_losses = []
        epoch_temporal_fused_losses = []
        epoch_temporal_alignments = []
        epoch_temporal_target_strengths = []
        epoch_temporal_transfer_means = []
        epoch_temporal_transfer_stds = []
        epoch_gradient_cosines = []
        epoch_gradient_conflicts = []
        epoch_gradient_raw_ratios = []
        epoch_gradient_applied_ratios = []
        epoch_gradient_warmup_scales = []
        epoch_temporal_gate_sum = None
        epoch_temporal_gate_min = None
        epoch_temporal_gate_max = None
        epoch_temporal_entropy_sum = None
        epoch_temporal_max_sum = None
        epoch_final_rescue_sum = None
        epoch_final_damage_sum = None
        epoch_final_disagreement_sum = None
        epoch_proposal_rescue_sum = None
        epoch_proposal_damage_sum = None
        epoch_temporal_residual_sum = None
        epoch_temporal_sample_count = 0
        epoch_temporal_batch_count = 0

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()

            gama = float(getattr(config, 'condition_loss_weight', 0.05))

            with torch.cuda.amp.autocast(enabled=True):
                r = np.random.rand(1)
                if r < config.cutmix_prob:
                    lam = np.random.beta(config.beta, config.beta)
                    rand_index = torch.randperm(batch_verb.size()[0]).cuda()
                    target_o_a = batch_obj
                    target_o_b = batch_obj[rand_index]
                    target_v_a = batch_verb
                    target_v_b = batch_verb[rand_index]
                    target_a_a = batch_target
                    target_a_b = batch_target[rand_index]
                    # label adjustment-new combinations
                    target_all_a_c = target_v_a * len(obj2idx) + target_o_b
                    target_all_a_d = target_v_b * len(obj2idx) + target_o_a

                    bbx1, bby1, bbx2, bby2 = rand_bbox(batch_img.size(), lam)
                    batch_img[:, :, :, bbx1:bbx2, bby1:bby2] = batch_img[rand_index, :, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_img.size()[-1] * batch_img.size()[-2]))
                    model_outputs = model(batch_img)
                    (
                        base_outputs,
                        global_pair_scores,
                        temporal_proposal_scores,
                        temporal_corrected_scores,
                        temporal_gate_weights,
                        temporal_transfer_logits,
                        temporal_transfer_weights,
                        temporal_pair_residual,
                        _temporal_component_feedback_scores,
                    ) = unpack_c2c_training_outputs(model_outputs)
                    p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = base_outputs

                    # component loss
                    loss_verb = Loss_fn(p_v * config.cosine_scale, target_v_a) * lam + Loss_fn(
                        p_v * config.cosine_scale, target_v_b) * (1.0 - lam)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, target_o_a) * lam + Loss_fn(p_o * config.cosine_scale,
                                                                                              target_o_b) * (1.0 - lam)

                    # train only
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = p_pair_v[:, train_v_inds, train_o_inds] + p_pair_o[:, train_v_inds, train_o_inds]

                    loss_com_train = Loss_fn(pred_com_train * config.cosine_scale, target_a_a) * lam + Loss_fn(
                        pred_com_train * config.cosine_scale, target_a_b) * (1.0 - lam)
                    # extend to unseen world
                    pred_com_all = (p_pair_v + p_pair_o).reshape(batch_verb.size()[0], -1)
                    loss_com_all = Loss_fn(pred_com_all * config.cosine_scale, target_all_a_c) + Loss_fn(
                        pred_com_all * config.cosine_scale, target_all_a_d)

                    loss_com = loss_com_train + loss_com_all * gama
                    temporal_target_a = target_a_a
                    temporal_target_b = target_a_b
                    temporal_object_a = target_o_a
                    temporal_object_b = target_o_b
                    temporal_target_mix = float(lam)

                    # hsic loss
                    obj_y = lam * F.one_hot(target_o_a.view(-1, 1), len(obj2idx))[:, 0] + (1.0 - lam) * F.one_hot(
                        target_o_b.view(-1, 1), len(obj2idx))[:, 0]
                    verb_y = lam * F.one_hot(target_v_a.view(-1, 1), len(attr2idx))[:, 0] + (1.0 - lam) * F.one_hot(
                        target_v_b.view(-1, 1), len(attr2idx))[:, 0]
                    vid_feat = vid_feat.mean(-1)
                    loss_hsic_v = hsic_normalized_cca(vid_feat, v_feat, 20) \
                                  - hsic_normalized_cca(v_feat, verb_y.float(), 20)
                    loss_hsic_o = hsic_normalized_cca(vid_feat, o_feat, 20) \
                                  - hsic_normalized_cca(o_feat, obj_y.float(), 20)
                    n_c = v_feat.shape[-1]
                    loss_hsic_vo = hsic_normalized_cca(v_feat[:, :int(n_c * 0.5)], o_feat[:, :int(n_c * 0.5)], 20)
                    loss_hsic = loss_hsic_v + loss_hsic_o + loss_hsic_vo

                    # condition loss
                    loss_con_train = torch.tensor([0.0]).cuda()

                else:
                    model_outputs = model(batch_img)
                    (
                        base_outputs,
                        global_pair_scores,
                        temporal_proposal_scores,
                        temporal_corrected_scores,
                        temporal_gate_weights,
                        temporal_transfer_logits,
                        temporal_transfer_weights,
                        temporal_pair_residual,
                        _temporal_component_feedback_scores,
                    ) = unpack_c2c_training_outputs(model_outputs)
                    p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = base_outputs
                    # component loss
                    loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                    loss_com_both = Loss_fn(pred_com_train * config.cosine_scale, batch_target)

                    loss_com = loss_com_both
                    temporal_target_a = batch_target
                    temporal_target_b = None
                    temporal_object_a = batch_obj
                    temporal_object_b = None
                    temporal_target_mix = 1.0

                    # hsic loss
                    obj_y = F.one_hot(batch_obj.view(-1, 1), len(obj2idx))[:, 0]
                    verb_y = F.one_hot(batch_verb.view(-1, 1), len(attr2idx))[:, 0]
                    vid_feat = vid_feat.mean(-1)
                    loss_hsic_v = hsic_normalized_cca(vid_feat, v_feat, 20) \
                                  - hsic_normalized_cca(v_feat, verb_y.float(), 20)
                    loss_hsic_o = hsic_normalized_cca(vid_feat, o_feat, 20) \
                                  - hsic_normalized_cca(o_feat, obj_y.float(), 20)
                    n_c = v_feat.shape[-1]
                    loss_hsic_vo = hsic_normalized_cca(v_feat[:, :int(n_c * 0.5)], o_feat[:, :int(n_c * 0.5)], 20)
                    loss_hsic = loss_hsic_v + loss_hsic_o + loss_hsic_vo


                    # condition loss
                    p_o_con_v_mean = p_o_con_v.mean(0)
                    p_v_con_o_mean = p_v_con_o.mean(0)
                    #
                    loss_on_v = Loss_fn(p_o_con_v_mean, v_o_on_v)
                    loss_on_o = Loss_fn(p_v_con_o_mean.permute(1, 0), v_o_on_o.permute(1, 0))
                    loss_con_train = loss_on_o + loss_on_v

                zero = loss_com.new_zeros(())
                loss_temporal_boost = zero
                loss_temporal_fused = zero
                temporal_residual_alignment = zero
                temporal_residual_target_strength = zero
                temporal_final_rescue = zero
                temporal_final_damage = zero
                temporal_final_disagreement = zero
                temporal_proposal_rescue = zero
                temporal_proposal_damage = zero
                if temporal_corrected_scores is not None:
                    if i == config.epoch_start and bid == 0:
                        resolved_object_a = train_o_inds.index_select(
                            0,
                            temporal_target_a,
                        )
                        if not torch.equal(
                                resolved_object_a,
                                temporal_object_a):
                            raise ValueError(
                                "Training pair targets and object targets use "
                                "inconsistent indexing."
                            )
                        if temporal_target_b is not None:
                            resolved_object_b = train_o_inds.index_select(
                                0,
                                temporal_target_b,
                            )
                            if not torch.equal(
                                    resolved_object_b,
                                    temporal_object_b):
                                raise ValueError(
                                    "CutMix pair targets and object targets "
                                    "use inconsistent indexing."
                                )
                    global_train_scores = global_pair_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    proposal_train_scores = temporal_proposal_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    corrected_train_scores = temporal_corrected_scores[
                        :, train_v_inds, train_o_inds
                    ]
                    (
                        loss_temporal_boost,
                        temporal_residual_alignment,
                        temporal_residual_target_strength,
                    ) = temporal_residual_boost_loss(
                        global_train_scores,
                        proposal_train_scores,
                        temporal_target_a,
                        temporal_target_b,
                        temporal_target_mix,
                        config.cosine_scale,
                    )
                    loss_temporal_fused = mixed_pair_cross_entropy(
                        corrected_train_scores,
                        temporal_target_a,
                        temporal_target_b,
                        temporal_target_mix,
                        config.cosine_scale,
                    )

                    final_a = temporal_change_statistics(
                        global_train_scores,
                        corrected_train_scores,
                        temporal_target_a,
                    )
                    proposal_a = temporal_change_statistics(
                        global_train_scores,
                        proposal_train_scores,
                        temporal_target_a,
                    )
                    if temporal_target_b is None:
                        final_b = final_a
                        proposal_b = proposal_a
                    else:
                        final_b = temporal_change_statistics(
                            global_train_scores,
                            corrected_train_scores,
                            temporal_target_b,
                        )
                        proposal_b = temporal_change_statistics(
                            global_train_scores,
                            proposal_train_scores,
                            temporal_target_b,
                        )
                    (
                        temporal_final_rescue,
                        temporal_final_damage,
                        temporal_final_disagreement,
                    ) = tuple(
                        temporal_target_mix * value_a
                        + (1.0 - temporal_target_mix) * value_b
                        for value_a, value_b in zip(final_a, final_b)
                    )
                    (
                        temporal_proposal_rescue,
                        temporal_proposal_damage,
                        _temporal_proposal_disagreement,
                    ) = tuple(
                        temporal_target_mix * value_a
                        + (1.0 - temporal_target_mix) * value_b
                        for value_a, value_b in zip(proposal_a, proposal_b)
                    )

                if temporal_pair_residual is not None:
                    temporal_residual_batch_sum = (
                        temporal_pair_residual.detach().float().abs().mean(dim=1)
                    ).sum()
                else:
                    temporal_residual_batch_sum = None

                if temporal_gate_weights is not None:
                    gate_for_stats = temporal_gate_weights.detach().float()
                    temporal_gate_batch_sum = gate_for_stats.sum(dim=0)
                    temporal_gate_batch_min = gate_for_stats.amin(dim=0)
                    temporal_gate_batch_max = gate_for_stats.amax(dim=0)
                    temporal_gate_entropy_sum = -(
                        gate_for_stats
                        * gate_for_stats.clamp_min(1.0e-8).log()
                    ).sum(dim=1).sum()
                    temporal_gate_maximum_sum = (
                        gate_for_stats.max(dim=1).values.sum()
                    )
                    temporal_gate_batch_size = gate_for_stats.shape[0]
                else:
                    temporal_gate_batch_sum = None
                    temporal_gate_batch_min = None
                    temporal_gate_batch_max = None
                    temporal_gate_entropy_sum = None
                    temporal_gate_maximum_sum = None
                    temporal_gate_batch_size = 0

                loss_global_anchor = (
                    loss_com
                    + 0.2 * loss_obj
                    + gama * loss_con_train
                )
                loss_global_remainder = (
                    0.2 * loss_verb
                    + 0.1 * loss_hsic
                )
                loss_temporal = (
                    temporal_boost_weight * loss_temporal_boost
                    + temporal_fused_weight * loss_temporal_fused
                )
                unscaled_loss = (
                    loss_global_anchor
                    + loss_global_remainder
                    + loss_temporal
                )
                ensure_finite_training_loss(
                    unscaled_loss,
                    {
                        "composition": loss_com,
                        "verb": loss_verb,
                        "object": loss_obj,
                        "hsic": loss_hsic,
                        "condition": loss_con_train,
                        "temporal_boost": loss_temporal_boost,
                        "temporal_fused": loss_temporal_fused,
                    },
                    i + 1,
                    bid + 1,
                )
                accumulation_steps = float(
                    config.gradient_accumulation_steps
                )
                loss = unscaled_loss / accumulation_steps
                if temporal_shared_warmup_epochs > 0.0:
                    epoch_progress = i + (
                        float(bid + 1) / max(len(train_dataloader), 1)
                    )
                    temporal_shared_warmup_scale = min(
                        1.0,
                        max(
                            0.0,
                            epoch_progress / temporal_shared_warmup_epochs,
                        ),
                    )
                else:
                    temporal_shared_warmup_scale = 1.0

            gradient_stats = global_anchored_backward(
                loss_global_anchor / accumulation_steps,
                loss_global_remainder / accumulation_steps,
                loss_temporal / accumulation_steps,
                scaler,
                shared_video_parameters,
                temporal_shared_gradient_ratio,
                temporal_shared_warmup_scale,
            )

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                if temporal_branch_parameters:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            temporal_branch_parameters,
                            max_norm=temporal_gradient_clip,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as error:
                        raise FloatingPointError(
                            "Non-finite temporal-branch gradient at epoch "
                            "%d batch %d." % (i + 1, bid + 1)
                        ) from error
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())
            epoch_hsic_v_losses.append(loss_hsic_v.item())
            epoch_hsic_o_losses.append(loss_hsic_o.item())
            epoch_hsic_vo_losses.append(loss_hsic_vo.item())
            epoch_con_train_losses.append(loss_con_train.item())
            temporal_loss_detached = (
                loss_temporal_boost + loss_temporal_fused
            ).detach().float()
            epoch_temporal_boost_losses.append(
                loss_temporal_boost.detach().float().item()
            )
            epoch_temporal_fused_losses.append(
                loss_temporal_fused.detach().float().item()
            )
            epoch_temporal_alignments.append(
                temporal_residual_alignment.detach().float().item()
            )
            epoch_temporal_target_strengths.append(
                temporal_residual_target_strength.detach().float().item()
            )
            epoch_gradient_cosines.append(gradient_stats["cosine"])
            epoch_gradient_conflicts.append(gradient_stats["conflict"])
            epoch_gradient_raw_ratios.append(gradient_stats["raw_ratio"])
            epoch_gradient_applied_ratios.append(
                gradient_stats["applied_ratio"]
            )
            epoch_gradient_warmup_scales.append(gradient_stats["warmup"])
            if temporal_transfer_weights is not None:
                transfer_values = temporal_transfer_weights.detach().float()
                selected_transfer = transfer_values.gather(
                    1,
                    temporal_object_a.unsqueeze(1),
                ).squeeze(1)
                if temporal_object_b is not None:
                    selected_transfer_b = transfer_values.gather(
                        1,
                        temporal_object_b.unsqueeze(1),
                    ).squeeze(1)
                    selected_transfer = (
                        temporal_target_mix * selected_transfer
                        + (1.0 - temporal_target_mix) * selected_transfer_b
                    )
                epoch_temporal_transfer_means.append(
                    selected_transfer.mean().item()
                )
                epoch_temporal_transfer_stds.append(
                    selected_transfer.std(unbiased=False).item()
                )
            batch_size = batch_verb.shape[0]
            if epoch_temporal_loss_sum is None:
                epoch_temporal_loss_sum = temporal_loss_detached.clone()
                epoch_final_rescue_sum = (
                    temporal_final_rescue.detach().float() * batch_size
                )
                epoch_final_damage_sum = (
                    temporal_final_damage.detach().float() * batch_size
                )
                epoch_final_disagreement_sum = (
                    temporal_final_disagreement.detach().float() * batch_size
                )
                epoch_proposal_rescue_sum = (
                    temporal_proposal_rescue.detach().float() * batch_size
                )
                epoch_proposal_damage_sum = (
                    temporal_proposal_damage.detach().float() * batch_size
                )
            else:
                epoch_temporal_loss_sum += temporal_loss_detached
                epoch_final_rescue_sum += (
                    temporal_final_rescue.detach().float() * batch_size
                )
                epoch_final_damage_sum += (
                    temporal_final_damage.detach().float() * batch_size
                )
                epoch_final_disagreement_sum += (
                    temporal_final_disagreement.detach().float() * batch_size
                )
                epoch_proposal_rescue_sum += (
                    temporal_proposal_rescue.detach().float() * batch_size
                )
                epoch_proposal_damage_sum += (
                    temporal_proposal_damage.detach().float() * batch_size
                )
            epoch_temporal_batch_count += 1
            if temporal_gate_batch_sum is not None:
                if epoch_temporal_gate_sum is None:
                    epoch_temporal_gate_sum = temporal_gate_batch_sum.clone()
                    epoch_temporal_gate_min = temporal_gate_batch_min.clone()
                    epoch_temporal_gate_max = temporal_gate_batch_max.clone()
                    epoch_temporal_entropy_sum = temporal_gate_entropy_sum.clone()
                    epoch_temporal_max_sum = temporal_gate_maximum_sum.clone()
                else:
                    epoch_temporal_gate_sum += temporal_gate_batch_sum
                    epoch_temporal_gate_min = torch.minimum(
                        epoch_temporal_gate_min,
                        temporal_gate_batch_min,
                    )
                    epoch_temporal_gate_max = torch.maximum(
                        epoch_temporal_gate_max,
                        temporal_gate_batch_max,
                    )
                    epoch_temporal_entropy_sum += temporal_gate_entropy_sum
                    epoch_temporal_max_sum += temporal_gate_maximum_sum
                if epoch_temporal_residual_sum is None:
                    epoch_temporal_residual_sum = (
                        temporal_residual_batch_sum.clone()
                    )
                else:
                    epoch_temporal_residual_sum += temporal_residual_batch_sum
                epoch_temporal_sample_count += temporal_gate_batch_size

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_seconds = time.perf_counter() - epoch_train_start
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        temporal_loss_mean = 0.0
        temporal_gate_entropy_mean = 0.0
        temporal_gate_maximum_mean = 0.0
        temporal_final_rescue_mean = 0.0
        temporal_final_damage_mean = 0.0
        temporal_final_disagreement_mean = 0.0
        temporal_proposal_rescue_mean = 0.0
        temporal_proposal_damage_mean = 0.0
        temporal_residual_mean = 0.0
        epoch_gate_mean = None
        epoch_gate_min = None
        epoch_gate_max = None
        if epoch_temporal_loss_sum is not None:
            temporal_loss_mean = (
                epoch_temporal_loss_sum / max(epoch_temporal_batch_count, 1)
            ).item()
        if epoch_temporal_sample_count > 0:
            sample_count = float(epoch_temporal_sample_count)
            epoch_gate_mean = (
                epoch_temporal_gate_sum / sample_count
            ).cpu()
            epoch_gate_min = epoch_temporal_gate_min.cpu()
            epoch_gate_max = epoch_temporal_gate_max.cpu()
            temporal_gate_entropy_mean = (
                epoch_temporal_entropy_sum / sample_count
            ).item()
            temporal_gate_maximum_mean = (
                epoch_temporal_max_sum / sample_count
            ).item()
            temporal_final_rescue_mean = (
                epoch_final_rescue_sum / sample_count
            ).item()
            temporal_final_damage_mean = (
                epoch_final_damage_sum / sample_count
            ).item()
            temporal_final_disagreement_mean = (
                epoch_final_disagreement_sum / sample_count
            ).item()
            temporal_proposal_rescue_mean = (
                epoch_proposal_rescue_sum / sample_count
            ).item()
            temporal_proposal_damage_mean = (
                epoch_proposal_damage_sum / sample_count
            ).item()
            temporal_residual_mean = (
                epoch_temporal_residual_sum / sample_count
            ).item()
        temporal_transfer_mean = (
            float(np.mean(epoch_temporal_transfer_means))
            if epoch_temporal_transfer_means else 0.0
        )
        temporal_transfer_std = (
            float(np.mean(epoch_temporal_transfer_stds))
            if epoch_temporal_transfer_stds else 0.0
        )
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(
            f"epoch {i + 1} train seconds {epoch_train_seconds:.2f}\n"
        )
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_v loss {np.mean(epoch_hsic_v_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_o loss {np.mean(epoch_hsic_o_losses)}\n")
        log_training.write(f"epoch {i + 1} hsic_vo loss {np.mean(epoch_hsic_vo_losses)}\n")
        log_training.write(f"epoch {i + 1} con_train loss {np.mean(epoch_con_train_losses)}\n")
        log_training.write(
            f"epoch {i + 1} temporal residual loss sum "
            f"{temporal_loss_mean}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal boost/fused loss "
            f"{np.mean(epoch_temporal_boost_losses):.6f}/"
            f"{np.mean(epoch_temporal_fused_losses):.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal residual alignment/target strength "
            f"{np.mean(epoch_temporal_alignments):.6f}/"
            f"{np.mean(epoch_temporal_target_strengths):.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} shared gradient cosine/conflict "
            f"{np.mean(epoch_gradient_cosines):.6f}/"
            f"{np.mean(epoch_gradient_conflicts):.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} shared temporal gradient raw/applied ratio "
            f"{np.mean(epoch_gradient_raw_ratios):.6f}/"
            f"{np.mean(epoch_gradient_applied_ratios):.6f} | "
            f"warmup {np.mean(epoch_gradient_warmup_scales):.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal residual scale mean/std "
            f"{temporal_transfer_mean:.6f}/"
            f"{temporal_transfer_std:.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal gate entropy "
            f"{temporal_gate_entropy_mean:.6f} | "
            f"max {temporal_gate_maximum_mean:.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal rescue/damage/net "
            f"{temporal_final_rescue_mean:.6f}/"
            f"{temporal_final_damage_mean:.6f}/"
            f"{temporal_final_rescue_mean - temporal_final_damage_mean:.6f} | "
            f"disagreement {temporal_final_disagreement_mean:.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal proposal rescue/damage/net "
            f"{temporal_proposal_rescue_mean:.6f}/"
            f"{temporal_proposal_damage_mean:.6f}/"
            f"{temporal_proposal_rescue_mean - temporal_proposal_damage_mean:.6f}\n"
        )
        log_training.write(
            f"epoch {i + 1} temporal pair residual abs "
            f"{temporal_residual_mean:.6f}\n"
        )
        if epoch_gate_mean is not None:
            gate_message = " | ".join(
                f"k{kernel}:{weight:.6f}"
                for kernel, weight in zip(
                    (3, 5, 7, 9), epoch_gate_mean.tolist()
                )
            )
            gate_message = (
                f"epoch {i + 1} temporal gate mean {gate_message}"
            )
            progress_bar.write(gate_message)
            log_training.write(gate_message + "\n")
            gate_range_message = " | ".join(
                f"k{kernel}:{minimum:.6f}/{mean:.6f}/{maximum:.6f}"
                for kernel, minimum, mean, maximum in zip(
                    (3, 5, 7, 9),
                    epoch_gate_min.tolist(),
                    epoch_gate_mean.tolist(),
                    epoch_gate_max.tolist(),
                )
            )
            log_training.write(
                f"epoch {i + 1} temporal gate min/mean/max "
                f"{gate_range_message}\n"
            )
        core_model = model.module if hasattr(model, 'module') else model
        if hasattr(core_model, 'c2c_temporal_composition_expert'):
            temporal_expert = core_model.c2c_temporal_composition_expert
            gate_prior = temporal_expert.gate_prior
            prior_message = "/".join(
                f"{value:.3f}" for value in gate_prior.detach().cpu().tolist()
            )
            fixed_weights = temporal_expert.fixed_gate_weights
            residual_scale = temporal_expert.gate_residual_scale
            if fixed_weights is not None:
                fixed_message = "/".join(
                    f"{value:.3f}"
                    for value in fixed_weights.detach().cpu().tolist()
                )
                gate_mode_message = f"fixed weights {fixed_message}"
            elif residual_scale is not None:
                factor = gate_prior.new_tensor(2.0 * residual_scale).exp()
                lower = gate_prior / (
                    gate_prior + (1.0 - gate_prior) * factor
                )
                upper = gate_prior * factor / (
                    1.0 - gate_prior + gate_prior * factor
                )
                bounds_message = "/".join(
                    f"{low:.3f}-{high:.3f}"
                    for low, high in zip(
                        lower.detach().cpu().tolist(),
                        upper.detach().cpu().tolist(),
                    )
                )
                gate_mode_message = (
                    f"bounded prior residual {residual_scale:.4f} "
                    f"bounds {bounds_message}"
                )
            else:
                gate_mode_message = (
                    f"adaptive temperature {temporal_expert.gate_temperature:.4f}"
                )
            branch_message = (
                f"epoch {i + 1} temporal branch settings "
                f"gate {gate_mode_message} | "
                f"feature residual "
                f"{core_model.temporal_composition_feature_residual:.4f} | "
                f"residual scale current/max "
                f"{float(core_model.c2c_global_anchored_temporal_residual.current_scale().detach().cpu()):.4f}/"
                f"{core_model.temporal_residual_max_scale:.4f} | "
                f"boost/fused weights "
                f"{temporal_boost_weight:.4f}/"
                f"{temporal_fused_weight:.4f} | "
                f"shared gradient cap "
                f"{temporal_shared_gradient_ratio:.4f} | "
                f"shared warmup epochs "
                f"{temporal_shared_warmup_epochs:.2f} | "
                f"gradient clip {temporal_gradient_clip:.2f} | "
                f"dynamic component-composed pair prototypes | "
                f"global-anchored asymmetric gradient protection | "
                f"initial prior {prior_message}"
            )
            log_training.write(branch_message + "\n")
        # log_training.write(f"epoch {i + 1} con_x loss {np.mean(epoch_con_x_losses)}\n")
        # log_training.write(f"epoch {i + 1} con_e loss {np.mean(epoch_con_e_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
        key_set = [
            "attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all",
            "best_seen", "best_unseen", "best_hm", "AUC",
        ] + TEMPORAL_DIAGNOSTIC_KEYS
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    key_set = [
                        "best_seen", "best_unseen", "AUC", "best_hm",
                        "attr_acc", "obj_acc",
                    ] + TEMPORAL_DIAGNOSTIC_KEYS
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()

        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))
