import re
import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO

MAX_SEQ_LENGTH = 300 
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE).eval()  
    return tokenizer, model

def process_sequence(tokenizer, model, raw_seq):
    try:

        clean_seq = raw_seq[:MAX_SEQ_LENGTH].upper()  
        clean_seq = re.sub(r"[UZOB]", "X", clean_seq)

        seq_length = min(len(clean_seq), MAX_SEQ_LENGTH)
        effective_max_length = seq_length + 2 
       
        inputs = tokenizer(
            " ".join(list(clean_seq)),
            add_special_tokens=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH + 2,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].long().to(DEVICE) 
        attention_mask = inputs["attention_mask"].float().to(DEVICE) 

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
       
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        embeddings = outputs.last_hidden_state[0][1:-1].cpu().numpy()

        
        if embeddings.shape[0] < MAX_SEQ_LENGTH:
            padding = np.zeros((MAX_SEQ_LENGTH - embeddings.shape[0], 1024))
            embeddings = np.vstack([embeddings, padding])
        return embeddings

    except Exception as e:
        return None


def main():
 
    tokenizer, model = initialize_model()

    input_path = "/home/ys/sunhuaiyang/predict/case_study/10.fasta"
    output_path = "/home/ys/sunhuaiyang/predict/case_study/feature/10T5_features.npy"

    sequences = [str(rec.seq) for rec in SeqIO.parse(input_path, "fasta")]
   
    features = []
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        
        
        if seq_len > 300 and DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        emb = process_sequence(tokenizer, model, seq)

        if emb is not None:
         
            assert emb.shape == (MAX_SEQ_LENGTH, 1024), f"维度异常: {emb.shape}"
            features.append(emb)

    np.save(output_path, np.array(features))

    exit()


if __name__ == "__main__":
    main()