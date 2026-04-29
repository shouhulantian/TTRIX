import os
import sys
import copy
import math
import pprint
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trix import tasks, util
from trix.models_entity import TRIX

separator = ">" * 30
line = "-" * 30


def _graph_has_time(graph):
    return getattr(graph, "target_edge_time", None) is not None


def _sample_batch_from_graph(graph, bs):
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]
    if _graph_has_time(graph):
        return torch.cat([
            graph.target_edge_index[:, edge_mask],
            graph.target_edge_type[edge_mask].unsqueeze(0),
            graph.target_edge_time[edge_mask].unsqueeze(0),
        ]).t()
    return torch.cat([
        graph.target_edge_index[:, edge_mask],
        graph.target_edge_type[edge_mask].unsqueeze(0),
    ]).t()


def multigraph_collator_temporal(batch, train_graphs):
    """Edge-count-weighted graph sampling: bigger graphs get proportionally
    more batches. Default; matches the existing pretrain_entity.py policy."""
    probs = torch.tensor([g.edge_index.shape[1] for g in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()
    graph = train_graphs[graph_id]
    return graph, _sample_batch_from_graph(graph, len(batch))


def multigraph_collator_temporal_equal(batch, train_graphs):
    """Uniform graph sampling: each graph picked with probability 1/N. Used
    when graphs are imbalanced (e.g. ICEWS14: 90k vs ICEWS0515: 461k) so the
    smaller graph isn't drowned out. Set ``train.sampling: equal`` to enable."""
    num_graphs = len(train_graphs)
    probs = torch.tensor([1.0 / num_graphs for _ in train_graphs]).float()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()
    graph = train_graphs[graph_id]
    return graph, _sample_batch_from_graph(graph, len(batch))


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None):

    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    # Sampler size proxy: concatenate per-graph target indices. The collator
    # ignores the resulting tensor's contents and resamples from a randomly
    # selected graph each step, so per-graph time/relation vocab gaps are fine.
    train_triplets = torch.cat([
        torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
        for g in train_data
    ])
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    sampling_mode = cfg.train.get("sampling", "proportional")
    if sampling_mode == "equal":
        collate = partial(multigraph_collator_temporal_equal, train_graphs=train_data)
    else:
        collate = partial(multigraph_collator_temporal, train_graphs=train_data)
    if util.get_rank() == 0:
        logger.warning(f"multigraph sampling: {sampling_mode}")
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler, collate_fn=collate,
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True,
        )
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
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
                train_graph, batch = batch
                # negative_sampling auto-detects 4-col batches and broadcasts
                # the time column over negatives. TRIX.forward also auto-detects.
                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_graph, batch)
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
                    logger.warning("binary cross entropy: %g" % loss)
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
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    all_metrics = []
    for graph_id, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):

        has_time = _graph_has_time(test_graph)
        if has_time:
            test_triplets = torch.cat([
                test_graph.target_edge_index,
                test_graph.target_edge_type.unsqueeze(0),
                test_graph.target_edge_time.unsqueeze(0),
            ]).t()
        else:
            test_triplets = torch.cat([
                test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0),
            ]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            # Keep `batch` 4-col so the model threads time through. Strict /
            # temporal filter helpers expect a 3-col triple_batch instead.
            if has_time:
                batch_time = batch[:, 3]
                triple_batch = batch[:, :3]
            else:
                batch_time = None
                triple_batch = batch
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if has_time and filters is not None and getattr(filters, "edge_time", None) is not None:
                t_mask, h_mask = tasks.temporal_strict_negative_mask(filters, triple_batch, batch_time)
            elif filters is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, triple_batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, triple_batch)
            pos_h_index, pos_t_index, pos_r_index = triple_batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                logger.warning("%s: %g" % (metric, score))
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            logger.warning(separator)

    avg_metric = sum(all_metrics) / len(all_metrics)
    return avg_metric


def _build_filtered_data_temporal(train_data, valid_data, test_data, device):
    """Per-graph filtering graph: union of train+valid+test target edges with
    matching edge_time when graphs carry time. Used by both temporal and strict
    negative-mask helpers."""
    out = []
    for trg, valg, testg in zip(train_data, valid_data, test_data):
        kwargs = dict(
            edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1),
            edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type]),
            num_nodes=trg.num_nodes,
        )
        if _graph_has_time(trg) and _graph_has_time(valg) and _graph_has_time(testg):
            kwargs["edge_time"] = torch.cat([
                trg.target_edge_time, valg.target_edge_time, testg.target_edge_time,
            ])
        out.append(Data(**kwargs).to(device))
    return out


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

    ds_class = cfg["dataset"]["class"]
    if ds_class in ("JointDataset", "JointTemporalDataset"):
        train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    else:
        train_data, valid_data, test_data = [dataset[0]], [dataset[1]], [dataset[2]]

    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in valid_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
            if _graph_has_time(graph):
                graph.target_edge_time = graph.target_edge_time[mask]
        short_valid = [sv.to(device) for sv in short_valid]

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

    model = TRIX(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_1_cfg=cfg.model.entity_model_1,
        entity_model_2_cfg=cfg.model.entity_model_2,
        alpha=cfg.model.get("alpha", 0.0),
        window_size=cfg.model.get("window_size", -1),
        window_mode=cfg.model.get("window_mode", "symmetric"),
    )

    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])

    model = model.to(device)

    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"

    filtered_data = _build_filtered_data_temporal(train_data, valid_data, test_data, device)

    train_and_validate(
        cfg, model, train_data,
        valid_data if "fast_test" not in cfg.train else short_valid,
        filtered_data=filtered_data, batch_per_epoch=cfg.train.batch_per_epoch,
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)
