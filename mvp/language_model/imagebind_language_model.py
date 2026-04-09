import os
from imagebind import data
import torch
from imagebind.models.imagebind_model import ImageBindModel
from imagebind.models.imagebind_model import ModalityType


def imagebind_huge(pretrained=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        root = os.path.join(os.path.expanduser('~'), ".cache/imagebind")
        path = os.path.join(os.path.expanduser('~'), ".cache/imagebind/imagebind_huge.pth")
        if not os.path.exists(path):
            print(
                "Downloading imagebind weights to ~/.cache/imagebind/imagebind_huge.pth ..."
            )
            os.makedirs(root, exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                path,
                progress=True,
            )

        model.load_state_dict(torch.load(path))

    return model


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def get_text_embedding(text):
    text_list = [text]

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device, os.path.join(os.path.dirname(__file__), "bpe/bpe_simple_vocab_16e6.txt.gz")),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings[ModalityType.TEXT].squeeze(0)
