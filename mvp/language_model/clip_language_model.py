import torch
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-L/14@336px", device=device)
model.eval()

def get_text_embedding(text):
    text_list = [text]
    text = clip.tokenize(text_list).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features.squeeze(0)
