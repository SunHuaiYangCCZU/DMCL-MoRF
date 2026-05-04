import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GCNConv, TransformerConv, GINEConv

from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")

# =======================
# 1) Residue feature utils
# =======================

AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20
}

def get_aa_onehot(res_name: str) -> np.ndarray:
    vec = np.zeros(21, dtype=np.float32)
    vec[AA_MAP.get(res_name, 20)] = 1.0
    return vec

# 5维理化性质：hydrophobicity(KD), charge, polarity, aromatic, volume
HYDRO = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}
CHARGE = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1}
POLAR = {"ARG", "ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR", "CYS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
VOLUME = {
    "ALA":  88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY":  60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER":  89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0,
}

def get_physchem(res_name: str) -> np.ndarray:
    hydro = HYDRO.get(res_name, 0.0) / 5.0       # roughly [-1, 1]
    charge = CHARGE.get(res_name, 0.0)           # -1/0/1
    polar = 1.0 if res_name in POLAR else 0.0
    aromatic = 1.0 if res_name in AROMATIC else 0.0
    vol = VOLUME.get(res_name, 0.0) / 230.0      # roughly [0, 1]
    return np.array([hydro, charge, polar, aromatic, vol], dtype=np.float32)

# =======================
# 2) Geometry: dihedral + RBF
# =======================

def dihedral(p0, p1, p2, p3) -> float:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    b0 = -(p1 - p0)
    b1 = (p2 - p1)
    b2 = (p3 - p2)

    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))

def safe_sin_cos(angle):
    if angle is None:
        return 0.0, 0.0
    return float(np.sin(angle)), float(np.cos(angle))

def rbf_expand(dist: np.ndarray, D_count=16, cutoff=8.0) -> np.ndarray:
    """
    dist: [E,]
    return: [E, D_count]
    """
    centers = np.linspace(0.0, cutoff, D_count, dtype=np.float32)
    # width: spacing
    width = (cutoff / (D_count - 1) + 1e-6)
    gamma = 1.0 / (width ** 2)
    dist = dist.astype(np.float32).reshape(-1, 1)
    return np.exp(-gamma * (dist - centers.reshape(1, -1)) ** 2).astype(np.float32)

def gaussian_edge_weight(dist: np.ndarray, sigma=2.0) -> np.ndarray:
    dist = dist.astype(np.float32)
    return np.exp(-(dist ** 2) / (2 * (sigma ** 2))).astype(np.float32)

# =======================
# 3) PDB -> Graph (with edge_attr)
# =======================

def pdb_to_graph(pdb_file, d_threshold=8.0, rbf_dim=16):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

    model = next(structure.get_models())

    coords, onehots, phys, bfac, angle_feats = [], [], [], [], []

    for chain in model:
        res_list = []
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue
            resname = res.get_resname()
            if resname not in AA_MAP:
                resname = "UNK"
            res_list.append((res, resname))

        for i, (res, resname) in enumerate(res_list):
            ca = res["CA"].get_coord()
            coords.append(ca)
            onehots.append(get_aa_onehot(resname))
            phys.append(get_physchem(resname))

            b = float(res["CA"].get_bfactor()) if "CA" in res else 0.0
            bfac.append(b / 100.0)  # scale

            phi, psi = None, None
            try:
                if i > 0:
                    prev_res, _ = res_list[i - 1]
                    if ("C" in prev_res) and ("N" in res) and ("CA" in res) and ("C" in res):
                        phi = dihedral(prev_res["C"].get_coord(),
                                       res["N"].get_coord(),
                                       res["CA"].get_coord(),
                                       res["C"].get_coord())
                if i < len(res_list) - 1:
                    next_res, _ = res_list[i + 1]
                    if ("N" in res) and ("CA" in res) and ("C" in res) and ("N" in next_res):
                        psi = dihedral(res["N"].get_coord(),
                                       res["CA"].get_coord(),
                                       res["C"].get_coord(),
                                       next_res["N"].get_coord())
            except Exception:
                phi, psi = None, None

            sphi, cphi = safe_sin_cos(phi)
            spsi, cpsi = safe_sin_cos(psi)
            angle_feats.append([sphi, cphi, spsi, cpsi])

    if len(coords) == 0:
        return None

    coords = np.asarray(coords, dtype=np.float32)                # [N,3]
    onehots = np.asarray(onehots, dtype=np.float32)              # [N,21]
    phys = np.asarray(phys, dtype=np.float32)                    # [N,5]
    angle_feats = np.asarray(angle_feats, dtype=np.float32)      # [N,4]
    bfac = np.asarray(bfac, dtype=np.float32).reshape(-1, 1)     # [N,1]

    # edges by CA-CA distance
    dist_mat = distance_matrix(coords, coords)
    src, dst = np.where((dist_mat < d_threshold) & (dist_mat > 0))
    if len(src) == 0:
        # 避免空边图导致某些conv不稳定
        src = np.array([0], dtype=np.int64)
        dst = np.array([0], dtype=np.int64)

    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # environment node feats: degree + dist_to_centroid
    N = coords.shape[0]
    deg = np.bincount(src, minlength=N).astype(np.float32).reshape(-1, 1)
    deg = deg / 50.0
    centroid = coords.mean(axis=0, keepdims=True)
    dist2c = np.linalg.norm(coords - centroid, axis=1).astype(np.float32).reshape(-1, 1)
    dist2c = dist2c / 30.0

    # final node x: 21 + 5 + 4 + 2 + 1 = 33
    x = np.concatenate([onehots, phys, angle_feats, deg, dist2c, bfac], axis=1).astype(np.float32)

    # ---- edge_attr (RBF distance) + edge_weight (gaussian) ----
    edge_dist = dist_mat[src, dst].astype(np.float32)  # [E,]
    edge_attr = rbf_expand(edge_dist, D_count=rbf_dim, cutoff=d_threshold)  # [E, rbf_dim]
    edge_weight = gaussian_edge_weight(edge_dist, sigma=2.0)               # [E,]

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float)
    )
    return data

# =======================
# 4) Edge-aware multi-branch encoder
# =======================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class MultiGraphEncoderEdgeAware(nn.Module):
    """
    三分支：
    - GCNConv + edge_weight（利用距离权重）
    - GINEConv + edge_attr（强边特征聚合）
    - TransformerConv + edge_attr（注意力 + 边特征）
    """
    def __init__(self, in_feats=33, hidden_feats=512, num_layers=3, max_len=300,
                 rbf_dim=16, heads=8, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.hidden_feats = hidden_feats
        self.rbf_dim = rbf_dim
        self.dropout = dropout

        # ---- GCN branch (edge_weight) ----
        self.gcns = nn.ModuleList([
            GCNConv(in_feats if i == 0 else hidden_feats, hidden_feats)
            for i in range(num_layers)
        ])

        # ---- GINE branch (edge_attr) ----
        self.gines = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_feats if i == 0 else hidden_feats
            nn_gine = nn.Sequential(
                nn.Linear(in_c, hidden_feats),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_feats, hidden_feats),
            )
            self.gines.append(GINEConv(nn=nn_gine, edge_dim=rbf_dim))

        # ---- Transformer branch (edge_attr) ----
        self.trans = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_feats if i == 0 else hidden_feats
            # out_channels 是每个 head 的维度；concat=True => heads*out_channels = hidden_feats
            out_per_head = hidden_feats // heads
            self.trans.append(TransformerConv(
                in_channels=in_c,
                out_channels=out_per_head,
                heads=heads,
                concat=True,
                edge_dim=rbf_dim,
                dropout=dropout
            ))

        self.fusion = nn.Linear(hidden_feats * 3, hidden_feats)

    def forward(self, batch_data: Batch):
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = getattr(batch_data, "edge_attr", None)
        edge_weight = getattr(batch_data, "edge_weight", None)

        x_gcn = x
        x_gine = x
        x_tr = x

        for i in range(len(self.gcns)):
            x_gcn = self.gcns[i](x_gcn, edge_index, edge_weight=edge_weight)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)

            x_gine = self.gines[i](x_gine, edge_index, edge_attr=edge_attr)
            x_gine = F.relu(x_gine)
            x_gine = F.dropout(x_gine, p=self.dropout, training=self.training)

            x_tr = self.trans[i](x_tr, edge_index, edge_attr=edge_attr)
            x_tr = F.relu(x_tr)
            x_tr = F.dropout(x_tr, p=self.dropout, training=self.training)

        x_fused = torch.cat([x_gcn, x_gine, x_tr], dim=-1)
        x_fused = self.fusion(x_fused)

        out, mask = to_dense_batch(x_fused, batch_data.batch, max_num_nodes=self.max_len)
        return out  # [B, max_len, hidden_feats]

class Branch2(nn.Module):
    def __init__(self, in_feats=33, hidden_feats=512, num_layers=3, max_len=300,
                 rbf_dim=16, device='cpu'):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.in_feats = in_feats
        self.rbf_dim = rbf_dim

        self.encoder = MultiGraphEncoderEdgeAware(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            num_layers=num_layers,
            max_len=max_len,
            rbf_dim=rbf_dim
        )
        self.projection = nn.Linear(hidden_feats, 512)

    def forward(self, inp):
        """
        inp 支持：
        - list[str]：pdb路径列表
        - Batch/Data：直接喂图（便于你dummy测试）
        """
        if isinstance(inp, Batch):
            batch_data = inp.to(self.device)
        elif isinstance(inp, Data):
            batch_data = Batch.from_data_list([inp]).to(self.device)
        else:
            pdb_files = inp
            graph_list = []
            for path in pdb_files:
                g = pdb_to_graph(path, d_threshold=8.0, rbf_dim=self.rbf_dim)
                if g is not None:
                    graph_list.append(g)
                else:
                    # dummy graph
                    dummy_x = torch.zeros((1, self.in_feats), dtype=torch.float)
                    dummy_pos = torch.zeros((1, 3), dtype=torch.float)
                    dummy_edge = torch.zeros((2, 0), dtype=torch.long)
                    dummy_edge_attr = torch.zeros((0, self.rbf_dim), dtype=torch.float)
                    dummy_edge_weight = torch.zeros((0,), dtype=torch.float)
                    graph_list.append(Data(
                        x=dummy_x, pos=dummy_pos, edge_index=dummy_edge,
                        edge_attr=dummy_edge_attr, edge_weight=dummy_edge_weight
                    ))
            batch_data = Batch.from_data_list(graph_list).to(self.device)

        out = self.encoder(batch_data)   # [B, max_len, hidden]
        out = self.projection(out)       # [B, max_len, 512]
        return out

# =======================
# 5) Demo
# =======================
if __name__ == "__main__":
    in_feats = 33
    rbf_dim = 16
    model = Branch2(in_feats=in_feats, hidden_feats=64, max_len=100, rbf_dim=rbf_dim)

    # dummy edge-aware graphs
    data1 = Data(
        x=torch.randn(50, in_feats),
        pos=torch.randn(50, 3),
        edge_index=torch.randint(0, 50, (2, 120)),
        edge_attr=torch.randn(120, rbf_dim),
        edge_weight=torch.rand(120)
    )
    data2 = Data(
        x=torch.randn(80, in_feats),
        pos=torch.randn(80, 3),
        edge_index=torch.randint(0, 80, (2, 200)),
        edge_attr=torch.randn(200, rbf_dim),
        edge_weight=torch.rand(200)
    )
    batch_data = Batch.from_data_list([data1, data2])

    output = model(batch_data)
    print("输出 shape:", output.shape)  # [2, 100, 512]