import math

import torch
import torch_geometric
from torch import nn

from . import tasks, layers
from trix.base_nbfnet import BaseNBFNet


def precompute_trans_pe(num_time: int, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal positional encoding.

    Returns a ``(num_time, dim)`` tensor where::

        pe[t, 2i]   = sin(t / 10000^(2i/dim))
        pe[t, 2i+1] = cos(t / 10000^(2i/dim))

    This is the same form as ``precompute_trans_pe`` in
    ``alan_fitter/ultra/models.py`` (FITTER) and is used to initialise the
    learnable time embedding for ``tcomplx`` / ``tntcomplx`` message
    functions, so training starts from a meaningful periodic basis instead
    of a random Gaussian. ``RoPE2`` is parameter-free and does not need this.

    Each ``GeneralizedRelationalConv`` layer (in ``layers.py``) imports this
    function lazily at construction time to seed its ``self.time`` weight or
    ``self.time_pe`` buffer.
    """
    assert dim % 2 == 0, "sinusoidal PE requires even dim"
    pe = torch.zeros(num_time, dim)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float)
        * (-math.log(10000.0) / dim)
    )
    position = torch.arange(0, num_time, dtype=torch.float).unsqueeze(1)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TRIX(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_1_cfg, entity_model_2_cfg,
                 alpha=0.0, window_size=-1, window_mode="symmetric"):
        """TRIX foundation model with optional TIGER-style local-window mixing.

        Args (new in this paper):
            alpha (float): mixing weight on the local-windowed score in
                ``alpha * local + (1 - alpha) * global``. Default ``0`` =
                global-only (vanilla TRIX behavior).
            window_size (int): Δt half-width. Edges with ``|edge_time - query_t|
                <= window_size`` are kept in the per-query local subgraph.
                Set to ``-1`` to disable the local pass.
            window_mode (str): "symmetric" -> ``[t - W, t + W]``;
                "causal" -> ``[t - W, t - 1]`` (no future edges, suitable for
                inductive eval).

        ``entity_model_2`` parameters are SHARED between the global and local
        passes (matches TIGER's design — same module called twice on different
        graphs). ``entity_model_1`` and ``relation_model`` run only on the
        global graph; their outputs (``relation_representations``) are reused
        for the local pass.
        """
        super(TRIX, self).__init__()

        self.relation_model = RelNet(**rel_model_cfg)
        self.entity_model_1 = EntityNet(**entity_model_1_cfg)
        self.entity_model_2 = EntityNet(**entity_model_2_cfg)

        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.window_mode = window_mode
        assert window_mode in ("symmetric", "causal")

    def forward(self, data, batch):
        # Global path (always run): produces full-graph score.
        relation_representations = self.relation_model(data, batch, self.entity_model_1)
        global_score = self.entity_model_2(data, relation_representations, batch)["score"]

        # Skip the local-window pass if disabled.
        if self.alpha == 0.0 or self.window_size < 0:
            return global_score

        # Per-query local subgraph path. Each query (h, t, r, t_q) gets a
        # subgraph filtered by edge_time, then runs the same entity_model_2
        # (parameter-shared with the global pass).
        if not (hasattr(data, "edge_time") and data.edge_time is not None):
            # No time info available -> can't build a windowed graph. Fall
            # back to global-only silently.
            return global_score
        if batch.shape[-1] < 4:
            # No query time on the batch -> can't build a per-query window.
            return global_score

        query_times = batch[:, 0, 3]
        entity_graph_t = self.generate_graph_t(data, query_times,
                                               self.window_size, self.window_mode)

        local_scores = []
        for i in range(len(query_times)):
            rel_repr_i = relation_representations[i:i + 1]
            batch_i = batch[i:i + 1]
            local_score_i = self.entity_model_2(entity_graph_t[i], rel_repr_i, batch_i)["score"]
            local_scores.append(local_score_i)
        local_score = torch.cat(local_scores, dim=0)

        return self.alpha * local_score + (1.0 - self.alpha) * global_score

    def generate_graph_t(self, data, query_times, window_size, mode="symmetric"):
        """Build a per-query local subgraph for entity_model_2.

        Filters ``data.edge_index`` / ``edge_type`` / ``edge_time`` by
        ``|edge_time - query_t| <= window_size`` (symmetric) or by
        ``edge_time in [t - W, t - 1]`` (causal).

        Returns a list of PyG Data objects, one per query in the batch. The
        relation graph is intentionally NOT rebuilt per query — the local
        pass reuses the global ``relation_representations`` (matches TIGER's
        transductive behaviour at alan_fitter/ultra/models.py:93).
        """
        edge_time = data.edge_time

        if mode == "causal":
            time_start = query_times - window_size - 1
            time_end = query_times - 1
        else:  # symmetric
            time_start = query_times - window_size
            time_end = query_times + window_size

        entity_graph_t = []
        for i in range(query_times.shape[0]):
            keep = (edge_time >= time_start[i]) & (edge_time <= time_end[i])
            graph_t = torch_geometric.data.Data(
                edge_index=data.edge_index[:, keep],
                edge_type=data.edge_type[keep],
                edge_time=edge_time[keep],
                num_nodes=data.num_nodes,
                num_relations=data.num_relations,
            )
            entity_graph_t.append(graph_t)
        return entity_graph_t

    def relation(self, data, batch):
        relation_representations = self.relation_model(data, batch, self.entity_model_1)
        return relation_representations


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
            
        self.node_mlp = torch.nn.Linear(self.dims[-1] * 2, self.dims[-1])

        self.layers_hh = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_hh.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_ht = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_ht.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_th = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_th.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_tt = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_tt.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def forward(self, data, batch, entity_model_1):
        rel_graph = data.relation_adj
        h_index = batch[:, 0, 2]

        node_representations = torch.ones(len(h_index), rel_graph["hh"].num_relations, self.layers_hh[0].input_dim).to(h_index.device)

        batch_size = len(h_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)
        
        boundary = torch.zeros(batch_size, rel_graph["hh"].num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (rel_graph["hh"].num_nodes, rel_graph["hh"].num_nodes)
        edge_weight_hh = torch.ones(rel_graph["hh"].num_edges, device=h_index.device)
        edge_weight_ht = torch.ones(rel_graph["ht"].num_edges, device=h_index.device)
        edge_weight_th = torch.ones(rel_graph["th"].num_edges, device=h_index.device)
        edge_weight_tt = torch.ones(rel_graph["tt"].num_edges, device=h_index.device)

        hiddens = []
        layer_input = boundary

        for i in range(len(self.layers_hh)):
            self.layers_hh[i].relation = node_representations
            self.layers_ht[i].relation = node_representations
            self.layers_th[i].relation = node_representations
            self.layers_tt[i].relation = node_representations

            hidden_hh = self.layers_hh[i](layer_input, query, boundary, rel_graph["hh"].edge_index, rel_graph["hh"].edge_type, size, edge_weight_hh)
            hidden_ht = self.layers_ht[i](layer_input, query, boundary, rel_graph["ht"].edge_index, rel_graph["ht"].edge_type, size, edge_weight_ht)
            hidden_th = self.layers_th[i](layer_input, query, boundary, rel_graph["th"].edge_index, rel_graph["th"].edge_type, size, edge_weight_th)
            hidden_tt = self.layers_tt[i](layer_input, query, boundary, rel_graph["tt"].edge_index, rel_graph["tt"].edge_type, size, edge_weight_tt)

            hidden = hidden_hh + hidden_ht + hidden_th + hidden_tt
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)

            layer_input = hidden
            relation_representations = layer_input
            
            if i == 0:
                node_representations = self.node_mlp(entity_model_1(data, relation_representations, batch)["feature"]).reshape(len(h_index), rel_graph["hh"].num_relations, -1)


        node_query = query.unsqueeze(1).expand(-1, rel_graph["hh"].num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]
     
        return output
    

class EntityNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1,
                 num_time=1, time_dependent=False, project_times=False, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        # Stored so the layer construction below can pass them to the conv layers,
        # and forward() can decide whether time threading is needed.
        self.num_time = num_time
        self.time_dependent = time_dependent
        self.project_times = project_times

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True,
                    num_time=num_time, time_dependent=time_dependent, project_times=project_times)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)


    def bellmanford(self, data, h_index, r_index, separate_grad=False,
                    time_type=None, query_time=None):
        batch_size = len(r_index)
        # Per-batch index used to gather query head's evolving state per layer
        # (used by RoPE2_decay_q for query-conditioned RoPE-attention gating).
        batch_idx = torch.arange(batch_size, device=h_index.device)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            # time_type / query_time / query_head_state are only consulted by the time-aware message
            # functions (RoPE2, RoPE2_decay, RoPE2_decay_q, tcomplx, tntcomplx); other layers ignore them.
            # query_head_state = current state of the query head h_q at this layer; for RoPE2_decay_q
            # this is the entity-side component of the LLM-attention-style query representation.
            query_head_state = layer_input[batch_idx, h_index]   # (B, dim)
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight,
                           time_type=time_type, query_time=query_time,
                           query_head_state=query_head_state)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        # Batch is (B, 1+num_negs, K) where K=3 (h, t, r) for static datasets and
        # K=4 (h, t, r, time) for temporal datasets. The time column is consumed
        # only by time-aware message functions; otherwise it's stripped here.
        if batch.shape[-1] == 4:
            h_index, t_index, r_index, time_index = batch.unbind(-1)
        else:
            h_index, t_index, r_index = batch.unbind(-1)
            time_index = None

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # Time-aware kwargs for the layers: edge_time on the message graph, and the
        # per-batch query time. Negatives in the batch share the same query time
        # as the positive, so we take time_index[:, 0].
        edge_time = getattr(data, "edge_time", None)
        query_time = time_index[:, 0] if time_index is not None else None

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                  time_type=edge_time, query_time=query_time)
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return {"score": score, "feature": output["node_feature"]}