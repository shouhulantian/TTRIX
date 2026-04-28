import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from typing import Tuple


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    # Learned-time message functions need a per-time embedding. Listed here so
    # the layer wires up self.time / self.time_projection only when needed.
    LEARNED_TIME_MSGS = ("tcomplx", "tntcomplx")

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=False, project_relations=False,
                 num_time=1, time_dependent=False, project_times=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations
        self.num_time = num_time
        self.time_dependent = time_dependent
        self.project_times = project_times

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )

        # Learned time embedding (tcomplx / tntcomplx). RoPE2 is parameter-free
        # and doesn't need this -- it computes Δt rotation from edge_time directly.
        # Initialization follows FITTER (alan_fitter): sinusoidal positional
        # encoding (Transformer-style) so the time embedding starts as a
        # meaningful periodic basis rather than random Gaussian, with two modes:
        #   * project_times=False: learnable nn.Embedding seeded with PE,
        #     can be fine-tuned (default for tcomplx / tntcomplx).
        #   * project_times=True: fixed PE buffer + small per-layer projection
        #     MLP, so the projection is the only learnable time-side weight.
        if self.message_func in self.LEARNED_TIME_MSGS:
            if time_dependent:
                # time embedding as a projection of the query relation embedding
                self.time_linear = nn.Linear(query_input_dim, num_time * input_dim)
            else:
                # Sinusoidal initialization (Transformer-style PE), defined in
                # models_entity.precompute_trans_pe -- imported lazily here to
                # avoid a circular import (models_entity itself imports layers).
                from .models_entity import precompute_trans_pe
                pe = precompute_trans_pe(num_time, input_dim)
                if not self.project_times:
                    # Learnable; initialise weight to sinusoidal PE.
                    self.time = nn.Embedding(num_time, input_dim)
                    with torch.no_grad():
                        self.time.weight.copy_(pe)
                else:
                    # Fixed sinusoidal PE + learnable MLP projection.
                    # `self.time` is unused in this mode; forward reads time_pe.
                    self.time = None
                    self.register_buffer("time_pe", pe)
                    self.time_projection = nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    )


    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None,
                time_type=None, query_time=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change:
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)

        # Construct per-layer time embedding only for learned-time message functions.
        # RoPE2 ignores `time` and uses query_time / time_type directly.
        time = None
        if self.message_func in self.LEARNED_TIME_MSGS and time_type is not None:
            if self.time_dependent:
                time = self.time_linear(query).view(batch_size, self.num_time, self.input_dim)
            elif not self.project_times:
                # Learnable embedding (sinusoidal-initialized in __init__).
                time = self.time.weight.expand(batch_size, -1, -1)
            else:
                # Fixed sinusoidal PE buffer projected by the per-layer MLP.
                time_query = self.time_pe.expand(batch_size, -1, -1)  # (B, T, dim)
                time = self.time_projection(time_query)

        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        # time / time_type / query_time are consulted by RoPE2 (parameter-free) and
        # tcomplx / tntcomplx (learned time embedding indexed by time_type).
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight,
                                time=time, time_type=time_type, query_time=query_time)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        # Always take the explicit message + aggregate path. The rspmm fast
        # kernel needs CUDA_HOME for JIT compilation (not set in this env),
        # and the time-aware message functions (RoPE2 / tcomplx / tntcomplx)
        # need per-edge logic the kernel can't express anyway. Matches
        # alan_fitter's `propagate` policy of routing every supported message
        # function through the explicit path.
        return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res

        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type, time=None, time_type=None, query_time=None):
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "tcomplx":
            # FITTER tComplEx: full complex multiplication of three complex
            # values (entity * relation * time) using their real / imag halves.
            if time is None or time_type is None:
                raise ValueError("tcomplx requires `time` (learned time embedding) "
                                 "and `time_type` (per-edge time index) — make sure "
                                 "data.edge_time is set and num_time is configured.")
            time_j = time.index_select(self.node_dim, time_type)  # (B, E, dim)
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            t_j_re, t_j_im = time_j.chunk(2, dim=-1)
            message_re = (x_j_re * r_j_re * t_j_re
                          - x_j_im * r_j_im * t_j_re
                          - x_j_im * r_j_re * t_j_im
                          - x_j_re * r_j_im * t_j_im)
            message_im = (x_j_im * r_j_re * t_j_re
                          + x_j_re * r_j_im * t_j_re
                          + x_j_re * r_j_re * t_j_im
                          - x_j_im * r_j_im * t_j_im)
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "tntcomplx":
            # FITTER TNTComplEx: time-modulated relation r' = r * (1 + t),
            # then standard complex multiplication of x and r'.
            if time is None or time_type is None:
                raise ValueError("tntcomplx requires `time` and `time_type` — see tcomplx docstring.")
            time_j = time.index_select(self.node_dim, time_type)  # (B, E, dim)
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            t_j_re, t_j_im = time_j.chunk(2, dim=-1)
            # rt = (re*re, im*re, re*im, im*im) of relation * time
            rt = r_j_re * t_j_re, r_j_im * t_j_re, r_j_re * t_j_im, r_j_im * t_j_im
            # full_rel = r + r*t = ((rt[0] - rt[3]) + r_j_re, (rt[1] + rt[2]) + r_j_im)
            full_rel_re = (rt[0] - rt[3]) + r_j_re
            full_rel_im = (rt[1] + rt[2]) + r_j_im
            message_re = x_j_re * full_rel_re - x_j_im * full_rel_im
            message_im = x_j_im * full_rel_re + x_j_re * full_rel_im
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "RoPE2":
            # rotate-style complex multiplication then per-edge Δt rotation.
            # query_time: (B,) -- per-query timestamps
            # time_type:  (E,) -- per-edge timestamps (must be on Data.edge_time)
            if query_time is None or time_type is None:
                raise ValueError("RoPE2 requires query_time and time_type; "
                                 "make sure data.edge_time is set and the model "
                                 "passes batch[..., 3] as query_time.")
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_im * r_j_re + x_j_re * r_j_im
            message = torch.cat([message_re, message_im], dim=-1)
            # time_delta shape (B, E): broadcasts against input_j shape (B, E, dim)
            time_delta = query_time.unsqueeze(1) - time_type.unsqueeze(0)
            message = self.rope_relative(message, time_delta.to(message.dtype))
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def rope_relative(self, x, d):
        """Apply RoPE rotation with relative time-distance d (no learned freq).

        Args:
          x: (..., dim) — the message tensor; dim must be even.
          d: tensor with x's leading shape minus the last dim. Treated as a
             relative timestep.

        Empty edge sets (num_edges=0) can occur in inductive subgraphs; handle
        that case by short-circuiting since view(..., -1, 2) is undefined for
        zero-element tensors.
        """
        if x.numel() == 0:
            return x
        dim = x.shape[-1]
        assert dim % 2 == 0, "RoPE2 requires even hidden dim"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim))
        theta = d.unsqueeze(-1) * inv_freq  # (..., dim/2)
        cos = torch.repeat_interleave(torch.cos(theta), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(theta), 2, dim=-1)

        x_view = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x_view[..., 0], x_view[..., 1]
        rotated = torch.stack((-x2, x1), dim=-1).reshape(*x_view.shape[:-2], -1)
        return x * cos + rotated * sin

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]

        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    