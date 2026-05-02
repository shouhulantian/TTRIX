import os
import sys
import math
import pprint
import copy
from itertools import islice

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trix import tasks, util
from trix.models_entity import TRIX

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # only used by the relation-similarity visualizer; not on main path
import numpy as np

separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return -1

    world_size = util.get_world_size()
    rank = util.get_rank()

    # Include target_edge_time as a 4th column if the dataset is temporal.
    # Downstream (negative_sampling, model.forward) auto-detect 4-column batches.
    if hasattr(train_data, "target_edge_time") and train_data.target_edge_time is not None:
        train_triplets = torch.cat([train_data.target_edge_index,
                                    train_data.target_edge_type.unsqueeze(0),
                                    train_data.target_edge_time.unsqueeze(0)]).t()
    else:
        train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("batch id: " + str(batch_id) + " binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    if best_epoch != -1:
        state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    else:
        state = torch.load("model_epoch_1.pth", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    util.synchronize()
    return best_epoch

@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    has_time = hasattr(test_data, 'target_edge_time') and test_data.target_edge_time is not None
    if has_time:
        test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0),
                                   test_data.target_edge_time.unsqueeze(0)]).t()
    else:
        test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []
    for batch in test_loader:
        # Keep `batch` as 4-col (h, t, r, time) so the model and all_negative
        # can thread time through to RoPE2 / tcomplx / tntcomplx layers. The
        # strict / temporal filter helpers expect a 3-col triple_batch instead.
        if has_time:
            batch_time = batch[:, 3]
            triple_batch = batch[:, :3]
        else:
            batch_time = None
            triple_batch = batch
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if has_time and filtered_data is not None and hasattr(filtered_data, 'edge_time'):
            t_mask, h_mask = tasks.temporal_strict_negative_mask(filtered_data, triple_batch, batch_time)
        elif filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, triple_batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, triple_batch)
        pos_h_index, pos_t_index, pos_r_index = triple_batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics

@torch.no_grad()
def test_rolling(cfg, model, test_data, device, logger, filtered_data=None, eval_mode="single_step"):
    """Per-timestep rolling-history eval matching RE-GCN/TKG-Forecasting-Eval protocol.

    eval_mode:
        "single_step" -- after scoring queries at time t, append GROUND-TRUTH test
                         events at t to the history graph for use at time t+1.
        "multi_step"  -- append MODEL TOP-1 PREDICTIONS at t (errors compound).

    The base message-passing graph is test_data.edge_index (which the dataset
    constructs as train+valid for chronological-split datasets). Test events
    accumulate ON TOP of this base.

    Optimization: relation_adj (TRIX's 4-subgraph hh/ht/th/tt) is built ONCE
    from the base train+valid graph and reused across all timesteps. Adding a
    few thousand test edges per timestep barely changes the relation co-
    occurrence pattern, so rebuilding relation_adj per timestep is wasteful
    and dominates wall-clock for datasets with many test timestamps. Set
    cfg.task.rebuild_relation_adj_per_step=True to override.
    """

    world_size = util.get_world_size()
    rank = util.get_rank()

    has_time = hasattr(test_data, 'target_edge_time') and test_data.target_edge_time is not None
    if not has_time:
        raise ValueError("test_rolling requires temporal data with target_edge_time")

    if isinstance(test_data.num_relations, torch.Tensor):
        num_rels_total = int(test_data.num_relations.item())
    else:
        num_rels_total = int(test_data.num_relations)
    num_rels_orig = num_rels_total // 2  # forward count; inverses use offset +num_rels_orig

    # Per-step relation_adj rebuild is expensive O(|R|^2) and barely changes
    # with a small number of new test edges; default skip = reuse base.
    rebuild_per_step = False
    if hasattr(cfg.task, 'get'):
        rebuild_per_step = cfg.task.get('rebuild_relation_adj_per_step', False)
    elif hasattr(cfg.task, 'rebuild_relation_adj_per_step'):
        rebuild_per_step = cfg.task.rebuild_relation_adj_per_step

    if rebuild_per_step:
        from trix.tasks import build_relation_graph
    else:
        # Pre-extract base relation_adj to clone into each timestep's Data
        base_relation_adj = test_data.relation_adj if hasattr(test_data, 'relation_adj') else None

    target_times = test_data.target_edge_time
    unique_times, _ = torch.sort(torch.unique(target_times))

    # Pull base history off-device (we need to rebuild relation_adj per timestep with CPU tensors)
    hist_edge_index = test_data.edge_index.detach().cpu()
    hist_edge_type = test_data.edge_type.detach().cpu()
    hist_edge_time = test_data.edge_time.detach().cpu()
    base_target_edge_index = test_data.target_edge_index.detach().cpu()
    base_target_edge_type = test_data.target_edge_type.detach().cpu()
    base_target_edge_time = test_data.target_edge_time.detach().cpu()

    all_rankings = []
    all_num_neg = []
    all_tail_rankings = []
    all_tail_num_neg = []

    model.eval()
    for t in unique_times.tolist():
        ts_mask_cpu = (base_target_edge_time == t)
        n_queries = int(ts_mask_cpu.sum().item())
        if n_queries == 0:
            continue
        if rank == 0:
            logger.warning(
                f"[{eval_mode}] timestep t={t}: {n_queries} queries, history edges={hist_edge_index.shape[1]}"
            )

        ts_data = Data(
            edge_index=hist_edge_index,
            edge_type=hist_edge_type,
            edge_time=hist_edge_time,
            target_edge_index=base_target_edge_index[:, ts_mask_cpu],
            target_edge_type=base_target_edge_type[ts_mask_cpu],
            target_edge_time=base_target_edge_time[ts_mask_cpu],
            num_relations=test_data.num_relations.cpu() if isinstance(test_data.num_relations, torch.Tensor) else test_data.num_relations,
            num_nodes=test_data.num_nodes,
        )
        if rebuild_per_step:
            ts_data = build_relation_graph(ts_data)
        elif base_relation_adj is not None:
            # Reuse the base train+valid relation_adj (cheap, defensible approximation)
            ts_data.relation_adj = base_relation_adj
        ts_data = ts_data.to(device)

        # Score queries at this timestep (mirrors test() inner loop)
        ts_triplets = torch.cat([
            ts_data.target_edge_index,
            ts_data.target_edge_type.unsqueeze(0),
            ts_data.target_edge_time.unsqueeze(0)
        ]).t()
        sampler = torch_data.DistributedSampler(ts_triplets, world_size, rank)
        ts_loader = torch_data.DataLoader(ts_triplets, cfg.train.batch_size, sampler=sampler)

        local_pos_h = []
        local_pos_t = []
        local_pos_r = []
        local_pred_t_top = []
        local_pred_h_top = []
        for batch in ts_loader:
            triple_batch = batch[:, :3]
            batch_time = batch[:, 3]
            t_batch, h_batch = tasks.all_negative(ts_data, batch)
            t_pred = model(ts_data, t_batch)
            h_pred = model(ts_data, h_batch)

            if filtered_data is not None and hasattr(filtered_data, 'edge_time'):
                t_mask, h_mask = tasks.temporal_strict_negative_mask(filtered_data, triple_batch, batch_time)
            elif filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(ts_data, triple_batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, triple_batch)

            pos_h_index, pos_t_index, pos_r_index = triple_batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

            all_rankings += [t_ranking, h_ranking]
            all_num_neg += [t_mask.sum(dim=-1), h_mask.sum(dim=-1)]
            all_tail_rankings += [t_ranking]
            all_tail_num_neg += [t_mask.sum(dim=-1)]

            if eval_mode == "multi_step":
                local_pos_h.append(pos_h_index)
                local_pos_t.append(pos_t_index)
                local_pos_r.append(pos_r_index)
                local_pred_t_top.append(t_pred.argmax(dim=-1))
                local_pred_h_top.append(h_pred.argmax(dim=-1))

        # Update history for next timestep
        if eval_mode == "single_step":
            new_fwd_edges = base_target_edge_index[:, ts_mask_cpu]
            new_fwd_etypes = base_target_edge_type[ts_mask_cpu]
            new_fwd_times = base_target_edge_time[ts_mask_cpu]
            new_edges_bi = torch.cat([new_fwd_edges, new_fwd_edges.flip(0)], dim=1)
            new_etypes_bi = torch.cat([new_fwd_etypes, new_fwd_etypes + num_rels_orig])
            new_times_bi = torch.cat([new_fwd_times, new_fwd_times])
            hist_edge_index = torch.cat([hist_edge_index, new_edges_bi], dim=1)
            hist_edge_type = torch.cat([hist_edge_type, new_etypes_bi])
            hist_edge_time = torch.cat([hist_edge_time, new_times_bi])
        elif eval_mode == "multi_step":
            l_h = torch.cat(local_pos_h) if local_pos_h else torch.empty(0, dtype=torch.long, device=device)
            l_t = torch.cat(local_pos_t) if local_pos_t else torch.empty(0, dtype=torch.long, device=device)
            l_r = torch.cat(local_pos_r) if local_pos_r else torch.empty(0, dtype=torch.long, device=device)
            l_pt = torch.cat(local_pred_t_top) if local_pred_t_top else torch.empty(0, dtype=torch.long, device=device)
            l_ph = torch.cat(local_pred_h_top) if local_pred_h_top else torch.empty(0, dtype=torch.long, device=device)

            if world_size > 1:
                # All-gather variable-size tensors: pad to per-rank max via tensor list
                local_n = torch.tensor([len(l_h)], device=device)
                sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
                dist.all_gather(sizes, local_n)
                max_n = int(torch.stack(sizes).max().item())
                def _pad(x):
                    if len(x) == max_n:
                        return x
                    pad = torch.zeros(max_n - len(x), dtype=x.dtype, device=x.device)
                    return torch.cat([x, pad])
                gh_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gt_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gr_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gpt_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gph_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                dist.all_gather(gh_list, _pad(l_h))
                dist.all_gather(gt_list, _pad(l_t))
                dist.all_gather(gr_list, _pad(l_r))
                dist.all_gather(gpt_list, _pad(l_pt))
                dist.all_gather(gph_list, _pad(l_ph))
                # Truncate each to per-rank size
                gh = torch.cat([gh_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gt = torch.cat([gt_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gr = torch.cat([gr_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gpt = torch.cat([gpt_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gph = torch.cat([gph_list[i][:int(sizes[i].item())] for i in range(world_size)])
            else:
                gh, gt, gr, gpt, gph = l_h, l_t, l_r, l_pt, l_ph

            tail_pred_edges = torch.stack([gh, gpt], dim=0).cpu()
            head_pred_edges = torch.stack([gph, gt], dim=0).cpu()
            new_edges_fwd = torch.cat([tail_pred_edges, head_pred_edges], dim=1)
            new_etypes_fwd = torch.cat([gr, gr]).cpu()
            new_times_fwd = torch.full((new_edges_fwd.shape[1],), t, dtype=torch.long)
            new_edges_bi = torch.cat([new_edges_fwd, new_edges_fwd.flip(0)], dim=1)
            new_etypes_bi = torch.cat([new_etypes_fwd, new_etypes_fwd + num_rels_orig])
            new_times_bi = torch.cat([new_times_fwd, new_times_fwd])
            hist_edge_index = torch.cat([hist_edge_index, new_edges_bi], dim=1)
            hist_edge_type = torch.cat([hist_edge_type, new_etypes_bi])
            hist_edge_time = torch.cat([hist_edge_time, new_times_bi])

    # Aggregate metrics (mirrors test()'s post-loop aggregation)
    ranking = torch.cat(all_rankings)
    num_negative = torch.cat(all_num_neg)
    tail_ranking = torch.cat(all_tail_rankings)
    num_tail_neg = torch.cat(all_tail_num_neg)

    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        logger.warning(line)
        logger.warning(f"[{eval_mode}] aggregated over {len(unique_times)} timesteps, {len(all_ranking)} total queries")
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name = metric.split("-")[0]
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking
                _num_neg = all_num_negative
                _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("[%s] %s: %g" % (eval_mode, metric, score))
            metrics[metric] = score
    return metrics


@torch.no_grad()
def cos_similarity(cfg, model, test_data, target_relation, device, logger, filtered_data=None, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    t_relations = []

    model.eval()
    for batch in test_loader:
        condition = (batch[:, 2] == target_relation)
        batch = batch[condition,:]
        if batch.size()[0] == 0:
            continue

        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_relation = model.relation(test_data, t_batch)
        t_relations.append(t_relation)

    average_relation = torch.mean(torch.cat(t_relations, axis=0), 0)
    
    cos = torch.nn.functional.cosine_similarity(average_relation[None,:,:].cpu(), average_relation[:,None,:].cpu(), dim=-1)

    plt.imshow(cos, cmap='gray', interpolation='nearest', vmin=-1, vmax=1) 
    cbar=plt.colorbar() 
    plt.tight_layout(pad=0.7)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig("./cos.pdf")

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)


    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = copy.deepcopy(valid_data)
        
        mask = torch.randperm(short_valid.target_edge_index.shape[1])[:num_val_edges]
        short_valid.target_edge_index = short_valid.target_edge_index[:, mask]
        short_valid.target_edge_type = short_valid.target_edge_type[mask]

    # Optional TIGER-style local-window mixing on entity_model_2.
    # Configs can declare alpha / window_size / window_mode at the top level
    # of cfg.model; defaults preserve the vanilla TRIX (global-only) behavior.
    model = TRIX(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_1_cfg=cfg.model.entity_model_1,
        entity_model_2_cfg=cfg.model.entity_model_2,
        alpha=cfg.model.get("alpha", 0.0),
        window_size=cfg.model.get("window_size", -1),
        window_mode=cfg.model.get("window_mode", "symmetric"),
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        # remap checkpoint keys if saved with old naming convention
        state_dict = state["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("entity_model_mini."):
                k = "entity_model_1." + k[len("entity_model_mini."):]
            elif k.startswith("entity_model."):
                k = "entity_model_2." + k[len("entity_model."):]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_kwargs = dict(edge_index=full_inference_edges, edge_type=full_inference_etypes,
                               num_nodes=test_data.num_nodes)
            if hasattr(test_data, 'edge_time') and test_data.edge_time is not None \
                    and hasattr(test_data, 'target_edge_time') and test_data.target_edge_time is not None:
                test_kwargs['edge_time'] = torch.cat([test_data.edge_time, test_data.target_edge_time])
            test_filtered_data = Data(**test_kwargs)

            # validation filtering graph: train edges + validation edges
            val_kwargs = dict(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type]),
            )
            if hasattr(train_data, 'edge_time') and train_data.edge_time is not None \
                    and hasattr(valid_data, 'target_edge_time') and valid_data.target_edge_time is not None:
                val_kwargs['edge_time'] = torch.cat([train_data.edge_time, valid_data.target_edge_time])
            val_filtered_data = Data(**val_kwargs)
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filter_kwargs = dict(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        # include timestamps for time-aware filtering if available
        if hasattr(dataset._data, 'target_edge_time') and dataset._data.target_edge_time is not None:
            filter_kwargs['edge_time'] = dataset._data.target_edge_time
        filtered_data = Data(**filter_kwargs)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    
    best_epoch = train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    if best_epoch == -1:
        state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        state_dict = state["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("entity_model_mini."):
                k = "entity_model_1." + k[len("entity_model_mini."):]
            elif k.startswith("entity_model."):
                k = "entity_model_2." + k[len("entity_model."):]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")

    # Dispatch on cfg.task.eval_mode (default: "static" = original behavior)
    eval_mode = cfg.task.get("eval_mode", "static") if hasattr(cfg.task, "get") else cfg.task.__dict__.get("eval_mode", "static")
    if eval_mode == "static":
        test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)
    elif eval_mode in ("single_step", "multi_step"):
        test_rolling(cfg, model, test_data, device=device, logger=logger,
                     filtered_data=test_filtered_data, eval_mode=eval_mode)
    elif eval_mode == "both":
        # Run static, single_step, and multi_step in sequence for direct comparison
        if util.get_rank() == 0:
            logger.warning("--- eval_mode=static ---")
        test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)
        if util.get_rank() == 0:
            logger.warning("--- eval_mode=single_step ---")
        test_rolling(cfg, model, test_data, device=device, logger=logger,
                     filtered_data=test_filtered_data, eval_mode="single_step")
        if util.get_rank() == 0:
            logger.warning("--- eval_mode=multi_step ---")
        test_rolling(cfg, model, test_data, device=device, logger=logger,
                     filtered_data=test_filtered_data, eval_mode="multi_step")
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")
