import torch
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_esm2, alphabet_esm2 = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_esm2 = model_esm2.to(device).eval()
batch_converter = alphabet_esm2.get_batch_converter()


class FastaDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (f"seq_{idx}", self.sequences[idx])


def collate_fn(batch, max_length=300):
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens[:, :max_length + 2]  
    return batch_labels, batch_strs, batch_tokens.to(device)


def process_batch(batch_tokens, max_length=300):
    with torch.no_grad():
        results = model_esm2(batch_tokens, repr_layers=[33])

    features = results["representations"][33]
    features = features[:, 1:-1, :]  

    seq_len = features.size(1)
    if seq_len < max_length:
        padding = torch.zeros((features.size(0), max_length - seq_len, features.size(2)),
                              device=device)
        features = torch.cat([features, padding], dim=1)
    else:
        features = features[:, :max_length, :]

    return features.detach().cpu().numpy()


def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) == 0:
            logging.warning(f"发现空序列: {record.id}")
            continue
        sequences.append(seq)
    return sequences


def main():
    file_path = '/home/ys/sunhuaiyang/predict/case_study/10.fasta'
    output_path = '/home/ys/sunhuaiyang/predict/case_study/feature/10esm_features.npy'

    sequences = read_fasta(file_path)
    if not sequences:
        raise ValueError("FASTA文件中未找到有效序列")

    dataset = FastaDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False,
                            collate_fn=lambda x: collate_fn(x))

    all_features = []
    for batch in dataloader:
        _, _, tokens = batch
        try:
            features = process_batch(tokens)
            all_features.append(features)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logging.error("GPU内存不足，请减小batch_size")
                raise
            else:
                raise

    final_features = np.concatenate(all_features, axis=0)
    np.save(output_path, final_features.astype(np.float32)) 
    logging.info(f"特征已保存至 {output_path} (形状: {final_features.shape})")


if __name__ == "__main__":
    main()