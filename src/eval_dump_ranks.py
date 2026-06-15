"""Standalone TRIX transductive eval that dumps per-query ranks to CSV.

Bundled separately from src/run_entity.py so the canonical train/eval
entrypoint stays untouched. Mirrors run_entity.test()'s filtering /
ranking logic so the CSV is directly comparable to the headline metrics
in the experiment doc.

CSV columns: h, t, r, time, side ('t' or 'h'), rank, num_neg
- One row per (query, side). The tail-side row carries the rank that
  results from masking everything except the gold tail; the head-side row
  is the symmetric head-corruption rank.
- Times are dataset-internal day offsets (TemporalICEWS14 uses
  day_ordinal - min_day).

Usage:
    python src/eval_dump_ranks.py -c <config.yaml> --gpus null
where <config.yaml> sets cfg.task.dump_ranks_path to the desired CSV
path. The script then runs eval-only (no training) and writes the CSV.
"""
import os
import sys
import csv
import pprint

import torch
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trix import tasks, util
from trix.models_entity import TRIX

separator = ">" * 30
line = "-" * 30


@torch.no_grad()
def test_dump(cfg, model, test_data, device, logger, filtered_data, dump_path):
    """Like run_entity.test() but also writes per-query rank CSV.

    Aggregate metrics still get logged so we can sanity-check that the
    numbers match the headline run before trusting the dump.
    """
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
    dump_rows = []
    for batch in test_loader:
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

        ph = pos_h_index.detach().cpu().tolist()
        pt = pos_t_index.detach().cpu().tolist()
        pr = pos_r_index.detach().cpu().tolist()
        bt = batch_time.detach().cpu().tolist() if has_time else [-1] * len(ph)
        tr = t_ranking.detach().cpu().tolist()
        hr = h_ranking.detach().cpu().tolist()
        tn = num_t_negative.detach().cpu().tolist()
        hn = num_h_negative.detach().cpu().tolist()
        for i in range(len(ph)):
            dump_rows.append((ph[i], pt[i], pr[i], bt[i], "t", tr[i], tn[i]))
            dump_rows.append((ph[i], pt[i], pr[i], bt[i], "h", hr[i], hn[i]))

    # Per-rank CSV (analysis script concatenates them). Naming:
    #   <dump_path>          when world_size==1
    #   <dump_path>.rank<R>  when world_size>1
    if world_size > 1:
        suffix = f".rank{rank}"
        out_path = dump_path + suffix
    else:
        out_path = dump_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["h", "t", "r", "time", "side", "rank", "num_neg"])
        w.writerows(dump_rows)
    logger.warning(f"[rank {rank}] wrote {len(dump_rows)} rows to {out_path}")

    # Sanity-check aggregate metrics (matches run_entity.test() behavior)
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
        import math
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

    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = TRIX(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_1_cfg=cfg.model.entity_model_1,
        entity_model_2_cfg=cfg.model.entity_model_2,
        alpha=cfg.model.get("alpha", 0.0),
        window_size=cfg.model.get("window_size", -1),
        window_mode=cfg.model.get("window_mode", "symmetric"),
    )

    if "checkpoint" not in cfg or cfg.checkpoint is None:
        raise ValueError("eval_dump_ranks.py needs cfg.checkpoint")

    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("entity_model_mini."):
            k = "entity_model_1." + k[len("entity_model_mini."):]
        elif k.startswith("entity_model."):
            k = "entity_model_2." + k[len("entity_model."):]
        new_state_dict[k] = v
    model_state = model.state_dict()
    filtered = {k: v for k, v in new_state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    missing = [k for k in model_state if k not in filtered]
    if missing and util.get_rank() == 0:
        logger.warning(f"missing ckpt keys: {missing[:8]} (total {len(missing)})")
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)

    # Transductive filtering graph: union of train+valid+test target edges
    # with timestamps for time-aware filtering. Matches run_entity.py's
    # transductive branch.
    filter_kwargs = dict(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
    )
    if hasattr(dataset._data, 'target_edge_time') and dataset._data.target_edge_time is not None:
        filter_kwargs['edge_time'] = dataset._data.target_edge_time
    filtered_data = Data(**filter_kwargs).to(device)

    dump_path = cfg.task.get("dump_ranks_path") if hasattr(cfg.task, "get") else cfg.task.__dict__.get("dump_ranks_path")
    if not dump_path:
        raise ValueError("cfg.task.dump_ranks_path is required")

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test (dump_ranks)")

    test_dump(cfg, model, test_data, device=device, logger=logger,
              filtered_data=filtered_data, dump_path=dump_path)
