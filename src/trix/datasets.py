import os
import csv
import shutil
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from trix.tasks import build_relation_graph


class GrailInductiveDataset(InMemoryDataset):

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, merge_valid_test=True):
        self.version = version
        assert version in ["v1", "v2", "v3", "v4"]

        # by default, most models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        # you'll need to delete the processed datasets then and re-run to cache a new dataset
        self.merge_valid_test = merge_valid_test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # creating fact graphs - those are graphs sent to a model, based on which we'll predict missing facts
        # also, those fact graphs will be used for filtered evaluation
        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]

        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # by default, SOTA models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        test_slice = slice(sum(num_samples[:3]), sum(num_samples)) if self.merge_valid_test else slice(sum(num_samples[:4]), sum(num_samples))
        
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice], num_relations=num_relations*2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice], num_relations=num_relations*2)
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice], num_relations=num_relations*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class FB15k237Inductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
    ]

    name = "IndFB15k237"

    def __init__(self, root, version):
        super().__init__(root, version)

class WN18RRInductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
    ]

    name = "IndWN18RR"

    def __init__(self, root, version):
        super().__init__(root, version)

class NELLInductive(GrailInductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
    ]
    name = "IndNELL"

    def __init__(self, root, version):
        super().__init__(root, version)


def FB15k237(root):
    dataset = RelLinkPredDataset(name="FB15k-237", root=root+"/fb15k237/")
    data = dataset.data
    train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type,
                        num_relations=dataset.num_relations)
    valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type,
                        num_relations=dataset.num_relations)
    test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type,
                        num_relations=dataset.num_relations)
    
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    return dataset

def WN18RR(root):
    dataset = WordNet18RR(root=root+"/wn18rr/")
    # convert wn18rr into the same format as fb15k-237
    data = dataset.data
    num_nodes = int(data.edge_index.max()) + 1
    num_relations = int(data.edge_type.max()) + 1
    edge_index = data.edge_index[:, data.train_mask]
    edge_type = data.edge_type[data.train_mask]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    edge_type = torch.cat([edge_type, edge_type + num_relations])
    train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.train_mask],
                        target_edge_type=data.edge_type[data.train_mask],
                        num_relations=num_relations*2)
    valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.val_mask],
                        target_edge_type=data.edge_type[data.val_mask],
                        num_relations=num_relations*2)
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.test_mask],
                        target_edge_type=data.edge_type[data.test_mask],
                        num_relations=num_relations*2)
    
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    dataset.num_relations = num_relations * 2
    return dataset


class TransductiveDataset(InMemoryDataset):

    delimiter = None
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"



class CoDEx(TransductiveDataset):

    name = "codex"
    urls = [
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/train.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/valid.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/test.txt",
    ]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.name, self.raw_dir)
            os.rename(download_path, path)


class CoDExSmall(CoDEx):
    """
    #node: 2034
    #edge: 36543
    #relation: 42
    """
    url = "https://zenodo.org/record/4281094/files/codex-s.tar.gz"
    md5 = "63cd8186fc2aeddc154e20cf4a10087e"
    name = "codex-s"

    def __init__(self, root):
        super(CoDExSmall, self).__init__(root=root, size='s')


class CoDExMedium(CoDEx):
    """
    #node: 17050
    #edge: 206205
    #relation: 51
    """
    url = "https://zenodo.org/record/4281094/files/codex-m.tar.gz"
    md5 = "43e561cfdca1c6ad9cc2f5b1ca4add76"
    name = "codex-m"
    def __init__(self, root):
        super(CoDExMedium, self).__init__(root=root, size='m')


class CoDExLarge(CoDEx):
    """
    #node: 77951
    #edge: 612437
    #relation: 69
    """
    url = "https://zenodo.org/record/4281094/files/codex-l.tar.gz"
    md5 = "9a10f4458c4bd2b16ef9b92b677e0d71"
    name = "codex-l"
    def __init__(self, root):
        super(CoDExLarge, self).__init__(root=root, size='l')


class NELL995(TransductiveDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # training set is made out of facts+train files, so we sum up their samples to build one training graph

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/test.txt",
    ]
    name = "nell995"

    @property
    def raw_file_names(self):
        return ["facts.txt", "train.txt", "valid.txt", "test.txt"]
    

    def process(self):
        train_files = self.raw_paths[:4]

        facts_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        train_results = self.load_file(train_files[1], facts_results["inv_entity_vocab"], facts_results["inv_rel_vocab"])
        valid_results = self.load_file(train_files[2], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[3], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        num_node = valid_results["num_node"]
        num_relations = train_results["num_relation"]

        train_triplets = facts_results["triplets"] + train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class ConceptNet100k(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
    ]
    name = "cnet100k"
    delimiter = "\t"


class DBpedia100k(TransductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
        ]
    name = "dbp100k"


class YAGO310(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
        ]
    name = "yago310"


class Hetionet(TransductiveDataset):

    urls = [
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        ]
    name = "hetionet"


class AristoV4(TransductiveDataset):

    url = "https://zenodo.org/record/5942560/files/aristo-v4.zip"

    name = "aristov4"
    delimiter = "\t"

    def download(self):
        download_path = download_url(self.url, self.raw_dir)
        extract_zip(download_path, self.raw_dir)
        os.unlink(download_path)
        for oldname, newname in zip(['train', 'valid', 'test'], self.raw_paths):
            os.rename(os.path.join(self.raw_dir, oldname), newname)


class SparserKG(TransductiveDataset):

    # 5 datasets based on FB/NELL/WD, introduced in https://github.com/THU-KEG/DacKGR
    # re-writing the loading function because dumps are in the format (h, t, r) while the standard is (h, r, t)

    url = "https://raw.githubusercontent.com/THU-KEG/DacKGR/master/data.zip"
    delimiter = "\t"
    base_name = "SparseKG"

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "processed")

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        for dsname in ['NELL23K', 'WD-singer', 'FB15K-237-10', 'FB15K-237-20', 'FB15K-237-50']:
            for oldname, newname in zip(['train.triples', 'dev.triples', 'test.triples'], self.raw_file_names):
                os.renames(os.path.join(base_path, "data", dsname, oldname), os.path.join(base_path, dsname, "raw", newname))
        shutil.rmtree(os.path.join(base_path, "data"))
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, v, r = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
class WDsinger(SparserKG):   
    name = "WD-singer"

class NELL23k(SparserKG):   
    name = "NELL23K"

class FB15k237_10(SparserKG):   
    name = "FB15K-237-10"

class FB15k237_20(SparserKG):   
    name = "FB15K-237-20"

class FB15k237_50(SparserKG):
    name = "FB15K-237-50"


class TemporalTransductiveDataset(TransductiveDataset):
    """Base class for temporal KG datasets with quadruples (h, r, t, timestamp).
    Loads data ignoring timestamps for structural TRIX, but stores timestamps
    for time-aware filtering during evaluation."""

    delimiter = "\t"
    store_timestamps = False  # subclasses set True for temporal filtering

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):
        triplets = []
        timestamps = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8-sig") as fin:
            for l in fin:
                parts = l.strip().split(self.delimiter)
                u, r, v = parts[0], parts[1], parts[2]
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u_id, r_id, v_id = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
                triplets.append((u_id, v_id, r_id))
                if len(parts) >= 4 and self.store_timestamps:
                    timestamps.append(parts[3])

        result = {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
        if self.store_timestamps:
            result["timestamps"] = timestamps
        return result


class ICEWS14(TemporalTransductiveDataset):
    """
    ICEWS14: 7128 entities, 230 relations, 365 timestamps, 90730 quadruples
    Random split (not chronological)
    """
    name = "icews14"

    def __init__(self, root, **kwargs):
        super(ICEWS14, self).__init__(root=root, **kwargs)


class ICEWS0515(TemporalTransductiveDataset):
    """
    ICEWS05-15: 10488 entities, 251 relations, 4017 timestamps, 461329 quadruples
    Random split (not chronological)
    """
    name = "icews0515"

    def __init__(self, root, **kwargs):
        super(ICEWS0515, self).__init__(root=root, **kwargs)


class TemporalICEWS14(TemporalTransductiveDataset):
    """ICEWS14 with timestamps stored for time-aware filtering."""
    name = "temporal_icews14"
    store_timestamps = True

    def __init__(self, root, **kwargs):
        super(TemporalICEWS14, self).__init__(root=root, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "icews14", "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "temporal_icews14", "processed")

    def _parse_date(self, date_str):
        """Convert YYYY-MM-DD to integer day offset."""
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.toordinal())

    def process(self):
        train_results = self.load_file(self.raw_paths[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(self.raw_paths[1],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(self.raw_paths[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])

        num_node = test_results["num_node"]
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        # Parse timestamps to integer day offsets
        all_dates = train_results["timestamps"] + valid_results["timestamps"] + test_results["timestamps"]
        date_ints = [self._parse_date(d) for d in all_dates]
        min_date = min(date_ints)

        train_times = torch.tensor([self._parse_date(d) - min_date for d in train_results["timestamps"]])
        valid_times = torch.tensor([self._parse_date(d) - min_date for d in valid_results["timestamps"]])
        test_times = torch.tensor([self._parse_date(d) - min_date for d in test_results["timestamps"]])

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        # Add inverse edges (timestamps are duplicated for inverse)
        train_edges_bi = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes_bi = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_times_bi = torch.cat([train_times, train_times])

        train_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2,
                          edge_time=train_times_bi, target_edge_time=train_times)
        valid_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes,
                          num_relations=num_relations * 2,
                          edge_time=train_times_bi, target_edge_time=valid_times)
        test_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes,
                         num_relations=num_relations * 2,
                         edge_time=train_times_bi, target_edge_time=test_times)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class TemporalICEWS0515(TemporalICEWS14):
    """ICEWS05-15 with timestamps stored for time-aware filtering."""
    name = "temporal_icews0515"

    @property
    def raw_dir(self):
        return os.path.join(self.root, "icews0515", "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "temporal_icews0515", "processed")


class InductiveDataset(InMemoryDataset):

    delimiter = None
    # some datasets (4 from Hamaguchi et al and Indigo) have validation set based off the train graph, not inference
    valid_on_inf = True  # 
    
    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, **kwargs):

        self.version = str(version)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "inf_valid.txt", "inf_test.txt"
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class IngramInductive(InductiveDataset):

    @property
    def raw_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "processed")
    

class FBIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/test.txt",
    ]
    name = "fb"


class WKIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/test.txt",
    ]
    name = "wk"

class NLIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/test.txt",
    ]
    name = "nl"


class ILPC2022(InductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/train.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_validation.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_test.txt",
    ]

    name = "ilpc2022"
    

class HM(InductiveDataset):
    # benchmarks from Hamaguchi et al and Indigo BM

    urls = [
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/train.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-graph.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/valid.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-fact.txt",
    ]

    name = "hm"
    versions = {
        '1k': "Hamaguchi-BM_both-1000",
        '3k': "Hamaguchi-BM_both-3000",
        '5k': "Hamaguchi-BM_both-5000",
        'indigo': "INDIGO-BM" 
    }
    # in 4 HM graphs, the validation set is based off the training graph, so we'll adjust the dataset creation accordingly
    valid_on_inf = False 

    def __init__(self, root, version, **kwargs):
        version = self.versions[version]
        super().__init__(root, version, **kwargs)

    # HM datasets are a bit weird: validation set (based off the train graph) has a few hundred new nodes, so we need a custom processing
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                          edge_type=train_fact_type, 
                          num_nodes=valid_res["num_node"],  # the only fix in this function
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class MTDEAInductive(InductiveDataset):

    valid_on_inf = False
    url = "https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip"
    base_name = "mtdea"

    def __init__(self, root, version, **kwargs):

        assert version in self.versions, f"unknown version {version} for {self.name}, available: {self.versions}"
        super().__init__(root, version, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "transductive_valid.txt", "inf_test.txt"
        ]

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        # unzip all datasets at once
        for dsname in ['FBNELL', 'Metafam', 'WikiTopics-MT1', 'WikiTopics-MT2', 'WikiTopics-MT3', 'WikiTopics-MT4']:
            cl = globals()[dsname.replace("-","")]
            versions = cl.versions
            for version in versions:
                for oldname, newname in zip(['train.txt', 'observe.txt', 'valid.txt', 'test.txt'], self.raw_file_names):
                    foldername = cl.prefix % version + "-trans" if "transductive" in newname else cl.prefix % version + "-ind"
                    os.renames(
                        os.path.join(base_path, "MTDEA_datasets", dsname, foldername, oldname), 
                        os.path.join(base_path, dsname, version, "raw", newname)
                    )
        shutil.rmtree(os.path.join(base_path, "MTDEA_datasets"))

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL and MT3:art, other datasets seem to be ok and share num_nodes/num_relations in the train/inference graph  
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))
        
        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    # special processes for MTDEA datasets for one particular fix in the validation set loading
    def process(self):
    
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class FBNELL(MTDEAInductive):

    name = "FBNELL"
    prefix = "%s"
    versions = ["FBNELL_v1"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(FBNELL, self).__init__(**kwargs)


class Metafam(MTDEAInductive):

    name = "Metafam"
    prefix = "%s"
    versions = ["Metafam"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(Metafam, self).__init__(**kwargs)


class WikiTopicsMT1(MTDEAInductive):

    name = "WikiTopics-MT1"
    prefix = "wikidata_%sv1"
    versions = ['mt', 'health', 'tax']

    def __init__(self, **kwargs):
        assert kwargs['version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)


class WikiTopicsMT2(MTDEAInductive):

    name = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['mt2', 'org', 'sci']

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)


class WikiTopicsMT3(MTDEAInductive):

    name = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['mt3', 'art', 'infra']

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)


class WikiTopicsMT4(MTDEAInductive):

    name = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['mt4', 'sci', 'health']

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)


# a joint dataset for pre-training TRIX on several graphs
class JointDataset(InMemoryDataset):

    datasets_map = {
        'FB15k237': FB15k237,
        'WN18RR': WN18RR,
        'CoDExSmall': CoDExSmall,
        'CoDExMedium': CoDExMedium,
        'CoDExLarge': CoDExLarge,
        'NELL995': NELL995,
        'ConceptNet100k': ConceptNet100k,
        'DBpedia100k': DBpedia100k,
        'YAGO310': YAGO310,
        'AristoV4': AristoV4,
    }

    def __init__(self, root, graphs, transform=None, pre_transform=build_relation_graph):
        self.graphs = [self.datasets_map[ds](root=root) for ds in graphs]
        self.num_graphs = len(graphs)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    
    def process(self):
        
        train_data = [g[0] for g in self.graphs]
        valid_data = [g[1] for g in self.graphs]
        test_data = [g[2] for g in self.graphs]
        # filter_data = [
        #     Data(edge_index=g.data.target_edge_index, edge_type=g.data.target_edge_type, num_nodes=g[0].num_nodes) for g in self.graphs
        # ]

        torch.save((train_data, valid_data, test_data), self.processed_paths[0])

class WikiTopics(InductiveDataset):

    valid_on_inf = False
    base_name = "WikiTopics"
    prefix = "wikidata_{}v2"
    versions = ["art", "award", "edu", "health", "infra", "sci", "sport", "tax"]

    def __init__(self, root, version, **kwargs):

        assert version in self.versions, f"unknown version {version} for {self.name}, available: {self.versions}"
        super().__init__(root, version, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.prefix.format(self.version))
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.prefix.format(self.version), "processed")
    
    @property
    def raw_file_names(self):
        return [
            "train.txt", "msg.txt", "valid.txt", "test.txt"
        ]

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL and MT3:art, other datasets seem to be ok and share num_nodes/num_relations in the train/inference graph  
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))
        
        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    # special processes for MTDEA datasets for one particular fix in the validation set loading
    def process(self):
    
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class WikiTopicsMeta(WikiTopics):

    base_name = "WikiTopics-MetaLearn2"
    versions = [
        "Run-1-Inf", "Run-1-InfSci", "Run-1-InfSciSpo", "Run-1-InfSciSpoTax", 
        "Run-2-Awa", "Run-2-AwaEdu", "Run-2-AwaEduTax", "Run-2-AwaEduTaxSpo",
        "Run-3-Spo", "Run-3-SpoInf", "Run-3-SpoInfEdu", "Run-3-SpoInfEduHea",
        "Run-4-Art", "Run-4-ArtAwa", "Run-4-ArtAwaHea", "Run-4-ArtAwaHeaInf",
        "Run-5-Hea", "Run-5-HeaSpo", "Run-5-HeaSpoTax", "Run-5-HeaSpoTaxArt"
    ]

    def __init__(self, **kwargs):
        super(WikiTopicsMeta, self).__init__(**kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.version)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "train.txt", "train.txt", "valid.txt", "valid.txt"
        ]


class ICEWS14to0515(InMemoryDataset):
    """Cross-dataset transfer: train on ICEWS14, zero-shot test on ICEWS05-15.
    Follows InductiveDataset pattern with separate train/inference graphs.
    Fully inductive: re-indexes all entities and relations from scratch."""

    delimiter = "\t"

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        self.train_path = os.path.join(root, "icews14", "raw")
        self.inf_path = os.path.join(root, "icews0515", "raw")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
        with open(triplet_file, "r", encoding="utf-8-sig") as fin:
            for l in fin:
                parts = l.strip().split(self.delimiter)
                u, r, v = parts[0], parts[1], parts[2]
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                triplets.append((inv_entity_vocab[u], inv_entity_vocab[v], inv_rel_vocab[r]))
        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    def process(self):
        # Load ICEWS14 as training graph (own vocab)
        train_res = self._load_file(
            os.path.join(self.train_path, "train.txt"),
            inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self._load_file(
            os.path.join(self.train_path, "valid.txt"),
            dict(train_res["inv_entity_vocab"]), dict(train_res["inv_rel_vocab"]))

        num_train_nodes = valid_res["num_node"]
        num_train_rels = valid_res["num_relation"]

        # Load ICEWS05-15 as inference graph (separate vocab)
        inf_train_res = self._load_file(
            os.path.join(self.inf_path, "train.txt"),
            inv_entity_vocab={}, inv_rel_vocab={})
        inf_test_res = self._load_file(
            os.path.join(self.inf_path, "test.txt"),
            dict(inf_train_res["inv_entity_vocab"]), dict(inf_train_res["inv_rel_vocab"]))

        inf_num_nodes = inf_test_res["num_node"]
        inf_num_rels = inf_test_res["num_relation"]

        # Build train graph (ICEWS14 train split)
        train_edges = train_res["triplets"]
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])
        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        # Validation targets from ICEWS14 valid split
        valid_edges = torch.tensor(valid_res["triplets"], dtype=torch.long)

        # Build inference graph (ICEWS05-15 train split as background graph)
        inf_graph_edges = inf_train_res["triplets"]
        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph_edges], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph_edges])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inf_num_rels])

        # Test targets from ICEWS05-15
        inf_test_edges = torch.tensor(inf_test_res["triplets"], dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_train_rels * 2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=valid_edges[:, :2].T,
                          target_edge_type=valid_edges[:, 2],
                          num_relations=num_train_rels * 2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes,
                         num_nodes=inf_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T,
                         target_edge_type=inf_test_edges[:, 2],
                         num_relations=inf_num_rels * 2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "icews14to0515", "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "icews14to0515", "processed")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    def __repr__(self):
        return "ICEWS14to0515()"


class GDELTIndT100Static(InductiveDataset):
    """Fully-inductive GDELT subset (INGRAM 4-file layout) with timestamps stripped.

    Raw layout under <root>/GDELTIndT_100/raw/:
        train.txt, valid.txt -- G_tr quadruples (h, r, t, date)
        msg.txt,   test.txt  -- G_inf quadruples (disjoint vocab from G_tr)

    Timestamps are dropped at load time. Valid uses the G_tr graph + G_tr valid
    targets (transductive valid, INGRAM convention); test uses the G_inf observed
    graph (msg.txt) + G_inf test targets. The two graph vocabs are disjoint, so
    inference truly tests inductive transfer.
    """

    name = "GDELTIndT_100"
    delimiter = "\t"
    valid_on_inf = False  # valid is held out from G_tr, not G_inf

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        InMemoryDataset.__init__(self, root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def load_file(self, triplet_file, inv_entity_vocab=None, inv_rel_vocab=None):
        inv_entity_vocab = dict(inv_entity_vocab) if inv_entity_vocab else {}
        inv_rel_vocab = dict(inv_rel_vocab) if inv_rel_vocab else {}
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
        triplets = []
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").split(self.delimiter)
                if len(parts) < 3:
                    continue
                u, r, v = parts[0], parts[1], parts[2]  # 4th column (timestamp) ignored
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt; entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt; entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt; rel_cnt += 1
                triplets.append((inv_entity_vocab[u], inv_entity_vocab[v], inv_rel_vocab[r]))
        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": len(inv_rel_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def process(self):
        # raw_paths order matches raw_file_names: [train, msg, valid, test]
        train_path, msg_path, valid_path, test_path = self.raw_paths[:4]

        train_res = self.load_file(train_path)
        valid_res = self.load_file(
            valid_path,
            inv_entity_vocab=train_res["inv_entity_vocab"],
            inv_rel_vocab=train_res["inv_rel_vocab"],
        )
        inf_res = self.load_file(msg_path)
        test_res = self.load_file(
            test_path,
            inv_entity_vocab=inf_res["inv_entity_vocab"],
            inv_rel_vocab=inf_res["inv_rel_vocab"],
        )

        # Use the EXTENDED vocab sizes so any train- or msg-only entities still get an id.
        num_train_nodes = valid_res["num_node"]
        num_train_rels = valid_res["num_relation"]
        num_inf_nodes = test_res["num_node"]
        num_inf_rels = test_res["num_relation"]

        train_edges = train_res["triplets"]
        valid_edges = valid_res["triplets"]
        inf_graph = inf_res["triplets"]
        test_edges = test_res["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])
        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edge_index = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edge_index = torch.cat([inf_edge_index, inf_edge_index.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + num_inf_rels])

        valid_t = torch.tensor(valid_edges, dtype=torch.long)
        test_t = torch.tensor(test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_train_rels * 2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=valid_t[:, :2].T, target_edge_type=valid_t[:, 2],
                          num_relations=num_train_rels * 2)
        test_data = Data(edge_index=inf_edge_index, edge_type=inf_etypes,
                         num_nodes=num_inf_nodes,
                         target_edge_index=test_t[:, :2].T, target_edge_type=test_t[:, 2],
                         num_relations=num_inf_rels * 2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed_static")

    @property
    def raw_file_names(self):
        return ["train.txt", "msg.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "GDELTIndT100Static()"


class InductiveTemporalDatasetINGRAM(InductiveDataset):
    """INGRAM-style fully-inductive temporal KG dataset (4-file layout).

    Raw layout under ``<root>/<name>/raw/``:
        train.txt -- G_tr training quadruples (h, r, t, date)
        msg.txt   -- G_inf observed graph (disjoint vocab from G_tr)
        valid.txt -- G_tr held-out validation queries (transductive valid)
        test.txt  -- G_inf held-out inductive test queries

    Each Data object exposes:
        edge_index, edge_type            -- message-passing graph (with inverse)
        edge_time                        -- per-edge timestamp (int day-ordinal offset)
        target_edge_index, target_edge_type, target_edge_time -- query edges
        num_relations                    -- 2 * num_unique_relations (forward + inverse)
        num_time                         -- max time index + 1, sized for RoPE2 freq table

    Temporal info (edge_time / target_edge_time / num_time) is available to:
      (a) message-passing layers when message_func='RoPE2' or similar (Δt rotation),
      (b) the time-aware ranking filter (temporal_strict_negative_mask).

    Subclasses set ``name`` (the directory name under root).
    """

    delimiter = "\t"
    valid_on_inf = False  # valid is held out from G_tr, not G_inf

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        InMemoryDataset.__init__(self, root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @staticmethod
    def _parse_date(date_str):
        from datetime import datetime
        return int(datetime.strptime(date_str, "%Y-%m-%d").toordinal())

    def load_file(self, triplet_file, inv_entity_vocab=None, inv_rel_vocab=None):
        inv_entity_vocab = dict(inv_entity_vocab) if inv_entity_vocab else {}
        inv_rel_vocab = dict(inv_rel_vocab) if inv_rel_vocab else {}
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
        triplets = []
        timestamps = []
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").split(self.delimiter)
                if len(parts) < 4:
                    continue
                u, r, v, ts = parts[0], parts[1], parts[2], parts[3]
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt; entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt; entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt; rel_cnt += 1
                triplets.append((inv_entity_vocab[u], inv_entity_vocab[v], inv_rel_vocab[r]))
                timestamps.append(ts)
        return {
            "triplets": triplets,
            "timestamps": timestamps,
            "num_node": len(inv_entity_vocab),
            "num_relation": len(inv_rel_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def process(self):
        train_path, msg_path, valid_path, test_path = self.raw_paths[:4]

        train_res = self.load_file(train_path)
        valid_res = self.load_file(valid_path,
                                   inv_entity_vocab=train_res["inv_entity_vocab"],
                                   inv_rel_vocab=train_res["inv_rel_vocab"])
        inf_res = self.load_file(msg_path)
        test_res = self.load_file(test_path,
                                  inv_entity_vocab=inf_res["inv_entity_vocab"],
                                  inv_rel_vocab=inf_res["inv_rel_vocab"])

        num_train_nodes = valid_res["num_node"]
        num_train_rels = valid_res["num_relation"]
        num_inf_nodes = test_res["num_node"]
        num_inf_rels = test_res["num_relation"]

        # Shared min-date so all four splits live in the same offset space.
        # Day-ordinal offsets are absolute; same encoding works across the
        # disjoint G_tr / G_inf vocabularies.
        all_dates = (train_res["timestamps"] + valid_res["timestamps"]
                     + inf_res["timestamps"] + test_res["timestamps"])
        min_ord = min(self._parse_date(d) for d in all_dates)
        max_ord = max(self._parse_date(d) for d in all_dates)
        num_time = int(max_ord - min_ord + 1)

        def to_t(stamps):
            return torch.tensor([self._parse_date(d) - min_ord for d in stamps], dtype=torch.long)

        train_t = to_t(train_res["timestamps"])
        valid_t_time = to_t(valid_res["timestamps"])
        inf_t = to_t(inf_res["timestamps"])
        test_t_time = to_t(test_res["timestamps"])

        # G_tr: train.txt is both the message-passing graph and the train targets
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_res["triplets"]], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_res["triplets"]])
        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])
        train_fact_time = torch.cat([train_t, train_t])

        # G_inf: msg.txt is the inference graph
        inf_edge_index = torch.tensor([[t[0], t[1]] for t in inf_res["triplets"]], dtype=torch.long).t()
        inf_edge_index_bi = torch.cat([inf_edge_index, inf_edge_index.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_res["triplets"]])
        inf_etypes_bi = torch.cat([inf_etypes, inf_etypes + num_inf_rels])
        inf_time_bi = torch.cat([inf_t, inf_t])

        valid_q = torch.tensor(valid_res["triplets"], dtype=torch.long)  # (n, 3) as (h, t, r)
        test_q = torch.tensor(test_res["triplets"], dtype=torch.long)

        # num_time is wrapped in a tensor so InMemoryDataset.collate keeps it
        # accessible as a 0-dim attribute (matches alan_fitter convention).
        num_time_t = torch.tensor([num_time])

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_train_rels * 2, num_time=num_time_t,
                          edge_time=train_fact_time, target_edge_time=train_t)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type,
                          num_nodes=num_train_nodes,
                          target_edge_index=valid_q[:, :2].T, target_edge_type=valid_q[:, 2],
                          num_relations=num_train_rels * 2, num_time=num_time_t,
                          edge_time=train_fact_time, target_edge_time=valid_t_time)
        test_data = Data(edge_index=inf_edge_index_bi, edge_type=inf_etypes_bi,
                         num_nodes=num_inf_nodes,
                         target_edge_index=test_q[:, :2].T, target_edge_type=test_q[:, 2],
                         num_relations=num_inf_rels * 2, num_time=num_time_t,
                         edge_time=inf_time_bi, target_edge_time=test_t_time)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed_temporal")

    @property
    def raw_file_names(self):
        return ["train.txt", "msg.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return f"{type(self).__name__}()"


class GDELTIndT100Temporal(InductiveTemporalDatasetINGRAM):
    """Fully-inductive GDELT split (V/R/T disjoint), temporal version."""
    name = "GDELTIndT_100"


class ICEWS14IndT100Temporal(InductiveTemporalDatasetINGRAM):
    """Fully-inductive ICEWS14 split (V/R/T disjoint), temporal version."""
    name = "ICEWS14IndT_100"


class ICEWS0515IndT100Temporal(InductiveTemporalDatasetINGRAM):
    """Fully-inductive ICEWS0515 split (V/R/T disjoint), temporal version."""
    name = "ICEWS0515IndT_100"


class TemporalGDELT(TemporalTransductiveDataset):
    """Transductive GDELT (full) with timestamps stored for time-aware filter
    and message-passing Δt rotation.
    """
    name = "temporal_gdelt"
    store_timestamps = True

    @property
    def raw_dir(self):
        return os.path.join(self.root, "GDELT", "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "temporal_gdelt", "processed")

    def _parse_date(self, date_str):
        # GDELT uses the same YYYY-MM-DD format as ICEWS
        from datetime import datetime
        return int(datetime.strptime(date_str, "%Y-%m-%d").toordinal())

    # Reuse TemporalICEWS14.process via inheritance through this minimal subclass:
    # behavior is identical, only raw_dir / processed_dir differ.
    process = TemporalICEWS14.process


class JointTemporalDataset(InMemoryDataset):
    """Joint temporal KG dataset with SHARED time vocabulary across graphs.

    Scans every source dataset's raw files, builds a single chronologically-
    sorted (date -> day-ordinal-offset) map, and re-loads each source under
    that shared map. Dates that appear in multiple sources (e.g. 2014-06-01
    in both ICEWS14 and ICEWS0515) get the same time index everywhere.

    Output format mirrors JointDataset: a 3-tuple
    ``(train_list, valid_list, test_list)`` where each list contains one PyG
    ``Data`` per source graph. This matches MultiGraphPretraining's contract.

    For Δt-only message functions (RoPE2) the shared encoding is mathematically
    equivalent to per-source indexing, but for absolute-time message functions
    (tcomplx, TTransE) it is required for correctness.
    """

    datasets_map = {
        "ICEWS14":   "TemporalICEWS14",
        "ICEWS0515": "TemporalICEWS0515",
        "GDELT":     "TemporalGDELT",
    }

    def __init__(self, root, graphs, transform=None, pre_transform=build_relation_graph, **kwargs):
        self.graphs_spec = list(graphs)
        self.num_graphs_spec = len(graphs)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        key = f"{self.num_graphs_spec}g_" + "_".join(self.graphs_spec)
        return os.path.join(self.root, "joint_t", key, "raw")

    @property
    def processed_dir(self):
        key = f"{self.num_graphs_spec}g_" + "_".join(self.graphs_spec)
        return os.path.join(self.root, "joint_t", key, "processed")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass  # source datasets handle their own downloads when instantiated

    @staticmethod
    def _parse_date(date_str):
        from datetime import datetime
        return int(datetime.strptime(date_str, "%Y-%m-%d").toordinal())

    @classmethod
    def _collect_dates(cls, raw_path):
        """Extract the 4th column (ISO date string) from a raw quadruple file."""
        dates = set()
        with open(raw_path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                dates.add(parts[3])
        return dates

    def process(self):
        import sys as _sys
        this_module = _sys.modules[__name__]
        sources = []
        for name in self.graphs_spec:
            cls_name = self.datasets_map[name]
            ds_cls = getattr(this_module, cls_name)
            sources.append(ds_cls(root=self.root))

        # Phase 1: build shared time vocab (day-ordinal offsets)
        all_dates = set()
        for src in sources:
            for raw_name in src.raw_file_names:
                raw_path = os.path.join(src.raw_dir, raw_name)
                if not os.path.exists(raw_path):
                    continue
                all_dates |= self._collect_dates(raw_path)
        date_ords = {d: self._parse_date(d) for d in all_dates}
        min_ord = min(date_ords.values())
        max_ord = max(date_ords.values())
        num_time = int(max_ord - min_ord + 1)
        num_time_t = torch.tensor([num_time])

        # Phase 2: rebuild each source's Data objects with the shared time map
        train_list, valid_list, test_list = [], [], []
        for src in sources:
            # Each source should already be a TemporalTransductiveDataset
            # whose load_file returns timestamps when store_timestamps=True.
            raw_paths = src.raw_paths[:3]  # train, valid, test
            train_res = src.load_file(raw_paths[0], inv_entity_vocab={}, inv_rel_vocab={})
            valid_res = src.load_file(raw_paths[1], train_res["inv_entity_vocab"], train_res["inv_rel_vocab"])
            test_res  = src.load_file(raw_paths[2], train_res["inv_entity_vocab"], train_res["inv_rel_vocab"])

            num_node = test_res["num_node"]
            num_rel = test_res["num_relation"]

            def to_t(stamps):
                return torch.tensor([date_ords[d] - min_ord for d in stamps], dtype=torch.long)

            train_t = to_t(train_res["timestamps"])
            valid_t_time = to_t(valid_res["timestamps"])
            test_t_time = to_t(test_res["timestamps"])

            train_q = train_res["triplets"]
            valid_q = valid_res["triplets"]
            test_q  = test_res["triplets"]

            train_target_edges = torch.tensor([[t[0], t[1]] for t in train_q], dtype=torch.long).t()
            train_target_etypes = torch.tensor([t[2] for t in train_q])
            valid_edges = torch.tensor([[t[0], t[1]] for t in valid_q], dtype=torch.long).t()
            valid_etypes = torch.tensor([t[2] for t in valid_q])
            test_edges  = torch.tensor([[t[0], t[1]] for t in test_q], dtype=torch.long).t()
            test_etypes = torch.tensor([t[2] for t in test_q])

            train_edges_bi = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
            train_etypes_bi = torch.cat([train_target_etypes, train_target_etypes + num_rel])
            train_times_bi = torch.cat([train_t, train_t])

            train_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                              target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                              num_relations=num_rel * 2, num_time=num_time_t,
                              edge_time=train_times_bi, target_edge_time=train_t)
            valid_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                              target_edge_index=valid_edges, target_edge_type=valid_etypes,
                              num_relations=num_rel * 2, num_time=num_time_t,
                              edge_time=train_times_bi, target_edge_time=valid_t_time)
            test_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                             target_edge_index=test_edges, target_edge_type=test_etypes,
                             num_relations=num_rel * 2, num_time=num_time_t,
                             edge_time=train_times_bi, target_edge_time=test_t_time)

            if self.pre_transform is not None:
                # build_relation_graph writes graph.relation_adj (TRIX's
                # 4-subgraph hh/ht/th/tt role-aware structure). Share across
                # splits since they all use train_data.edge_index as the
                # message graph.
                train_data = self.pre_transform(train_data)
                valid_data.relation_adj = train_data.relation_adj
                test_data.relation_adj = train_data.relation_adj

            train_list.append(train_data)
            valid_list.append(valid_data)
            test_list.append(test_data)

        torch.save((train_list, valid_list, test_list), self.processed_paths[0])

    def __repr__(self):
        return f"JointTemporalDataset(graphs={self.graphs_spec})"
