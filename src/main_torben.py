from einops import rearrange, repeat
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.data.utils import read_ts_from_fp
from src.visualization.time_series import plot_ts
from src.constants import READY_PATH, DATA_PATH


def l2norm(t):
    return F.normalize(t, dim=-1)


def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


class Network(nn.Module):
    def __init__(self, depth=5, hidden_dim=256):
        super(Network, self).__init__()
        self.proj = nn.Sequential(nn.Linear(2, hidden_dim), self.RMSNorm(hidden_dim), nn.SiLU())
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.silu1 = nn.SiLU()

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.silu2 = nn.SiLU()
        network = []
        for _ in range(depth):
            network.extend(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        self.RMSNorm(hidden_dim),
                        nn.SiLU(),
                        self._Attention(hidden_dim),
                    )
                ]
            )
        self.transformer = nn.ModuleList(network)
        self.final = nn.Linear(hidden_dim, 1)

    class RMSNorm(nn.Module):
        def __init__(self, dim, scale=True, normalize_dim=2):
            super().__init__()
            self.g = nn.Parameter(torch.ones(dim)) if scale else 1

            self.scale = scale
            self.normalize_dim = normalize_dim

        def forward(self, x):
            normalize_dim = self.normalize_dim
            scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
            return F.normalize(x, dim=normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)

    class _Attention(nn.Module):
        def __init__(self, dim, heads=4, dim_head=32, dropout=0.0, rmsnorm=True, mem_efficient=True, num_mem_kv=4):
            super().__init__()

            self.heads = heads
            hidden_dim = dim_head * heads

            self.norm = nn.LayerNorm(dim)

            self.dropout = dropout

            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

            self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

            self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        def flash_attn(self, q, k, v):
            # _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

            q, k, v = map(lambda t: t.contiguous(), (q, k, v))

            with torch.backends.cuda.sdp_kernel(True, True, True):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            return out

        def forward(self, x):
            b, h, c = x.shape
            x = self.norm(x)

            qkv = self.to_qkv(x).chunk(3, dim=-1)

            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

            q = q * self.q_scale
            k = k * self.k_scale

            mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
            # k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))
            k = torch.cat([mk, k], -2)
            v = torch.cat([mv, v], -2)

            out = self.flash_attn(q, k, v)

            out = rearrange(out, "b h n d -> b n (h d)")

            return self.to_out(out)

    def forward(self, x):
        hidden = self.proj(x)
        f = self.conv1(hidden.transpose(-1, -2)).transpose(-1, -2)
        f = self.ln1(f)
        f = self.silu1(f)
        f = self.conv2(f.transpose(-1, -2)).transpose(-1, -2)
        f = self.ln2(f)
        hidden = self.silu2(f) + hidden

        for layer in self.transformer:
            hidden = layer(hidden) + hidden
        return F.sigmoid(self.final(hidden).squeeze(-1))


def collate_fn(batch):
    data, labels = zip(*batch)

    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return data_padded, labels_padded


class HDF5Dataset(Dataset):
    def __init__(self, file_path, split, n_dates=64):
        assert split in ["train", "valid", "test"]
        self.file_path = file_path
        self.dataset_names = []
        self.split = split
        self.n_dates = n_dates

        if split == "train":
            self.group_name = ["fold1", "fold2", "fold3", "fold4"]
        elif split == "valid":
            self.group_name = ["fold5"]
        else:
            self.group_name = ["test"]

        self.file = h5py.File(self.file_path, "r")
        self.data = []
        self.metas = []
        # Load everything in memory (small so ok)
        for gn in self.group_name:
            group = self.file[gn]
            for _, item in group.items():
                self.data.append(item[...])
                self.metas.append(item.attrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ### ====== TORBEN SUGGESTIONS ======= ###
        # 1. don't set n_dates, use different lengths to augment the data (OD: would still set a max length)
        # 2. collate_fn will take care of padding
        # 3. drop randomly X and Y items on the left and right
        # 4. make sure that it does not cut between id_left_bound and id_right_bound
        # 5. You can also drop some parts of the sequence randomly (np.delete)
        # 6. For test set, need to augment it but deterministically. For instance use left half,
        #   right half, slider, ...
        # 7. For test set, need to make sure that the dataset is balanced (positive and negative)
        # (OD: would not zero pad test set)
        ### ================================= ###

        sequence = self.data[idx]  # Full sequence
        meta = self.metas[idx]

        # Get bounds of the sequence
        id_left_bound = meta["id_left_bound"]  # last date before date of invasion
        id_right_bound = meta["id_right_bound"]  # first date after UNOSAT analysis
        label_mask = np.zeros((len(sequence)))
        label_mask[id_left_bound + 1 : id_right_bound] = 1  # where the destruction might have happened

        # 1/2 chance of being positive
        label = np.random.randint(0, 2)

        # Get a random subsequence of length n_dates (data augmentation)
        # if label, then the subsequence must contain both id_left_bound and id_right_bound
        # otherwise, must be before id_left_bound
        # if training, set this randomly. If testing, set this deterministically using idx (but still shifting around)

        if label:
            min_idx = max(0, id_right_bound - self.n_dates + 1)  # at least one date after UNOSAT analysis
            max_idx = min(len(sequence) - self.n_dates, id_left_bound - 10)  # at least 10 dates before the war
        else:
            min_idx = 0
            max_idx = id_left_bound - self.n_dates

        for k, v in meta.items():
            print(k, v)
        print(label, id_left_bound, id_right_bound, min_idx, max_idx)
        if self.split == "train":
            start_idx = np.random.randint(min_idx, max_idx)
        else:
            start_idx = min_idx + (idx % 5) * (max_idx - min_idx) // 5
        end_idx = start_idx + self.n_dates

        sequence = sequence[start_idx:end_idx]
        label_mask = label_mask[start_idx:end_idx]

        # How many frames to drop from the left and right
        # if self.training:
        #     drop_left, drop_right = 0, 1
        #     if np.random.randint(0, 2) == 1:
        #         drop_left = np.random.randint(0, 24)
        #     if np.random.randint(0, 2) == 1:
        #         drop_right = np.random.randint(1, 24)

        #     if label == 1:
        #         drop_left = np.clip(drop_left, 0, id_left_bound)
        #         drop_right = np.clip(drop_right, 1, id_right_bound)

        #     sequence = sequence[drop_left:]
        #     sequence = sequence[:-drop_right]
        #     label_mask = label_mask[drop_left:]
        #     label_mask = label_mask[:-drop_right]

        # pose = np.arange(0,len(sequence))/len(sequence)
        # sequence = np.clip(sequence,-40,40)
        # sequence[:,0] = (sequence[:,0] - np.min(sequence[:,0]))/(np.max(sequence[:,0])-np.min(sequence[:,0]))
        # sequence[:,1] = (sequence[:,1] - np.min(sequence[:,1]))/(np.max(sequence[:,1])-np.min(sequence[:,1]))
        # sequence = np.concatenate([sequence,pose[...,None]],-1)

        # idx = torch.randperm(sequence.shape[0])
        data_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label_mask, dtype=torch.int64)

        return data_tensor, label_tensor


class Cherrys:
    def __init__(self, filepaths):
        self.cherrys = [read_ts_from_fp(fp) for fp in filepaths]
        print(f"🍒 Loaded {len(self.cherrys)} cherrys. 🍒")
        self.folder_preds = DATA_PATH / "tmp" / "cherry" / "preds"
        self.folder_preds.mkdir(exist_ok=True, parents=True)

    def inference(self, model):
        for i in range(len(self.cherrys)):
            cherry = self.cherrys[i].copy()
            cherry_ = torch.tensor(cherry.to_numpy()).cuda().unsqueeze(0).float()
            pred = model(cherry_)
            fp = self.folder_preds / f"cherry_{i}.png"
            self.plot_cherry(cherry, pred[0].cpu().detach().numpy(), fp)

    def plot_cherry(self, cherry, pred, fp):
        ax = plot_ts(cherry)
        ax2 = ax.twinx()
        ax2.plot(cherry.date, pred, label="pred", color="k")
        ax2.set_ylim([0, 1])
        ax2.legend()
        plt.savefig(fp, dpi=300)
        plt.close()


class Trainer:
    def __init__(self, filepath, cherrys_fp=None):
        # Model
        self.model = Network().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # self.clf_loss = nn.BCELoss()

        # Dataloader
        print("Loading data...", end=" ")
        self.train_dataloader = self.get_dataloader(filepath, split="train")
        self.valid_dataloader = self.get_dataloader(filepath, split="valid")
        self.test_dataloader = self.get_dataloader(filepath, split="test")
        print("Done!")

        self.cherrys = Cherrys(cherrys_fp) if cherrys_fp else None

    def get_dataloader(self, filepath, split):
        dset = HDF5Dataset(filepath, split=split)
        return DataLoader(
            dset,
            shuffle=split == "train",
            batch_size=256,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def compute_metrics(self, labels, label_preds):
        label_preds_bin = label_preds > 0.5
        d_metrics = {
            "f1": f1_score(labels, label_preds_bin),
            "precision": precision_score(labels, label_preds_bin),
            "recall": recall_score(labels, label_preds_bin),
            "accuracy": accuracy_score(labels, label_preds_bin),
        }
        return d_metrics

    def full_run(self, epochs=100):
        for epoch in range(epochs):
            print(f"Epoch {epoch} |", end=" ")
            self.train()
            self.inference(dataloader=self.valid_dataloader, prefix="valid_")
            if self.cherrys:
                self.cherrys.inference(self.model)

            if (epoch + 1) % 5 == 0:
                self.inference(dataloader=self.test_dataloader, prefix="test_")

    def train(self):
        self.model.train()

        losses = []
        labels = []
        label_preds = []
        for x, y in self.train_dataloader:
            self.optimizer.zero_grad()
            x = x.cuda()
            pred = self.model(x)
            mask = y.cuda()

            label = torch.max(mask, dim=-1)[0]
            labels.append(label.cpu().detach().numpy())

            label_pred = torch.max(pred, dim=-1)[0]
            label_preds.append(label_pred.cpu().detach().numpy())

            weight = torch.sum(mask, dim=-1) / len(mask[0])
            weight = 1 / weight
            weight = torch.nan_to_num(weight, 1, posinf=1).unsqueeze(-1)

            loss = torch.mean(weight * torch.square(pred - mask))

            # loss = clf_loss(label_pred,label.float())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        print(f"Loss={np.mean(losses):.5f}", end=" ")

        labels = np.concatenate(labels)
        label_preds = np.concatenate(label_preds)

        # compute metrics
        d_metrics = self.compute_metrics(labels, label_preds)
        print(", ".join([f"{k}={v:.2f}" for k, v in d_metrics.items()]), end="\t")

    def inference(self, dataloader, prefix=""):
        self.model.eval()

        losses = []
        labels = []
        label_preds = []
        for x, y in dataloader:
            x = x.cuda()
            pred = self.model(x)
            mask = y.cuda()

            label = torch.max(mask, dim=-1)[0]
            labels.append(label.cpu().detach().numpy())

            label_pred = torch.max(pred, dim=-1)[0]
            label_preds.append(label_pred.cpu().detach().numpy())

            weight = torch.sum(mask, dim=-1) / len(mask[0])
            weight = 1 / weight
            weight = torch.nan_to_num(weight, 1, posinf=1).unsqueeze(-1)

            loss = torch.mean(weight * torch.square(pred - mask))
            losses.append(loss.item())

        labels = np.concatenate(labels)
        label_preds = np.concatenate(label_preds)

        # compute metrics
        d_metrics = self.compute_metrics(labels, label_preds)
        print(", ".join([f"{prefix}{k}={v:.2f}" for k, v in d_metrics.items()]))


if __name__ == "__main__":
    extraction_strategy = "3x3"
    filepath = READY_PATH / f"dataset_{extraction_strategy}.h5"
    # filepath = ="/home/tpeters/Downloads/dataset_3x3.h5"
    print(f'Using "{extraction_strategy}" extraction strategy.')

    cherrys_base_folder = DATA_PATH / "tmp" / f"cherrys_{extraction_strategy}"
    cherry_folder = cherrys_base_folder / "easy_ts_xarray"
    cherry_folder_shifted = cherrys_base_folder / "easy_ts_xarray_shifted"
    cherry_folder_neg = cherrys_base_folder / "easy_ts_xarray_neg"
    cherrys_fp = []
    for folder in [cherry_folder, cherry_folder_shifted, cherry_folder_neg]:
        cherrys_fp += list(folder.glob("*.nc"))

    # cherrys_fp = None

    # trainer = Trainer(filepath, cherrys_fp)
    # trainer.full_run(epochs=100)

    split = "train"
    ds = HDF5Dataset(filepath, split=split)
    dl = DataLoader(
        ds,
        shuffle=split == "train",
        batch_size=256,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    item = next(iter(ds))
    print(item)
