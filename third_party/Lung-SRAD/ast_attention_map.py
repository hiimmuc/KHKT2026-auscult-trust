# Implementation adapted from https://github.com/MzeroMiko/VMamba/blob/main/analyze/attnmap.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial

from models.ast import ASTModel


def _strip_prefix(state_dict, prefix="module."):
    # Removes "module." 
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _filter_heads(state_dict):
    # Keep only backbone weights
    drop_prefixes = (
        "mlp_head.",          
        "classifier.",        
        "head.",              
        "fc.",                
    )
    return {k: v for k, v in state_dict.items() if not k.startswith(drop_prefixes)}


def load_model_only_weights(model, ckpt_path, device):
    """
    Loads only the model/backbone weights from a checkpoint that may contain
    model/classifier/optimizer/args/etc.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Print checkpoint keys 
    if isinstance(ckpt, dict):
        print("Checkpoint keys:", ckpt.keys())
    else:
        print("Checkpoint is not a dict")

    # Extract model state_dict
    if isinstance(ckpt, dict):

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
    
            is_tensor = any(hasattr(v, "shape") for v in ckpt.values())
            sd = ckpt if is_tensor else None
            if sd is None:
                raise ValueError("Could not find model weights in checkpoint dict.")
    else:
        sd = ckpt

    sd = _strip_prefix(sd, "module.")
    sd = _filter_heads(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded model-only weights.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    return model


def main_deit(model_ckpt=None, device=None, mode = None, head_index= None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32
    model = ASTModel(
        label_dim=4,
        fstride=10,           
        tstride=10,           
        input_fdim=128,
        input_tdim=798,       
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
        mix_beta=None
    ).to(device=device, dtype=dtype)

    if model_ckpt is not None:
        model_ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        print(model_ckpt['args'])
        model_ckpt = model_ckpt['model']
        missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        print(model)
        
    model.eval()

    vit = model.v if not hasattr(model, "module") else model.module.v

    attns = {}

    def attn_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)
        setattr(self, "__data__", attn.detach().cpu())
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    for i, blk in enumerate(vit.blocks):
        old_attn = blk.attn
        old_attn._orig_forward = old_attn.forward
        old_attn.forward = partial(attn_forward, old_attn)
        attns[str(i)] = old_attn
    print(attns.keys())

    x = np.load("/icbhi/spectrogram_data/batch13_spectrogram_label2.npy")
    print(x.shape)
    x = torch.from_numpy(x).to(device=device, dtype=dtype)
    x = x.unsqueeze(1)  # (B,1,T,F)
    x = F.interpolate(x, size=(798, 128), mode='bilinear', align_corners=False)
    x = x.squeeze(1)    # (B,798,128)

    with torch.no_grad():
        _ = model(x)
    

    out_maps = {}
    for k,mod in attns.items():
        saved_attn = getattr(mod, "__data__", None)
        if saved_attn is None:
            raise RuntimeError(f"No attention map saved in block {k}")
        if mode == 'head':
            out_maps[int(k)] = saved_attn[0][head_index].float()
            
        else:
            out_maps[int(k)] = saved_attn[0].float().mean(dim=0)
    return out_maps
        


def pairwise_cosine_similarity(x):
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    return x @ x.t()


def compute_cosine_similarity(seq_matrix):
    if isinstance(seq_matrix, np.ndarray):
        seq_matrix = torch.tensor(seq_matrix, dtype=torch.float32)
    if seq_matrix.dim() != 2:
        raise ValueError("Input must be 2D.")
    cos_mat = pairwise_cosine_similarity(seq_matrix)
    return cos_mat.mean().item()


if __name__ == "__main__":
    ckpt_path = "/pretrained_models/AST/model.pth"
    layers = 11
    mode = 'head'  # options include head for specific head or average for average for over all the heads
    head_index = 0 if mode == 'head' else None # head_index = 0 default
    attn_maps = main_deit(model_ckpt= ckpt_path, head_index = head_index, mode = mode)
    for i, attn_map in sorted(attn_maps.items()):
        print("attn_map shape:", attn_map.shape)
        np.save(f"/attention_maps/AST/attention_map_ast_{i}.npy", attn_map.numpy())



