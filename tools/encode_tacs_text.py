"""Create a fixed BioMedCLIP text embedding for the optional TACS router prior.

Install open_clip_torch first. The embedding is generated once and stored as
a small .pt file; the segmentation model does not run the text encoder.
"""
import argparse

import torch
import torch.nn.functional as F
import open_clip


MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_PROMPT = (
    "Lung tumor: focal PET uptake guided by CT anatomy and lesion boundaries, "
    "distinct from physiological uptake."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.eval()
    tokens = tokenizer([args.prompt])
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = F.normalize(embedding.float(), dim=-1).squeeze(0).cpu()
    torch.save({
        "embedding": embedding,
        "prompt": args.prompt,
        "model_id": MODEL_ID,
    }, args.output)
    print(f"saved embedding shape={tuple(embedding.shape)} to {args.output}")


if __name__ == "__main__":
    main()
