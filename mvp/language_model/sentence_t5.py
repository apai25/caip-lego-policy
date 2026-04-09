import torch
from sentence_transformers import SentenceTransformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('sentence-transformers/sentence-t5-large')
model.eval()
model.to(device)

def get_text_embedding(text):
    text_list = [text]

    with torch.no_grad():
        embeddings = model.encode(text_list)

    return torch.Tensor(embeddings).squeeze(0)
