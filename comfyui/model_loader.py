import os
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from diffusers.models import AutoencoderKLCogVideoX
import models

class LuminaVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_model_name": (["Alpha-VLLM/Lumina-Video-f24R960"], {
                    "default": "Alpha-VLLM/Lumina-Video-f24R960"
                }),
                "video_model_precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16"
                }),
                "text_encoder_name": (["google/gemma-2-2b"], {
                    "default": "google/gemma-2-2b"
                }),
                "vae_model_name": (["THUDM/CogVideoX-2b"], {
                    "default": "THUDM/CogVideoX-2b"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "TOKENIZER", "TEXT_ENCODER")
    RETURN_NAMES = ("model", "vae", "tokenizer", "text_encoder")
    FUNCTION = "load_models"
    CATEGORY = "Lumina-Video"

    def load_models(self, video_model_name, video_model_precision, text_encoder_name, vae_model_name):
        # 设置本地模型路径
        base_path = os.path.dirname(os.path.dirname(__file__))
        lumina_path = os.path.join(base_path, "models", "Lumina-Video")
        llm_path = os.path.join(base_path, "models", "LLM")
        
        # 确保目录存在
        os.makedirs(lumina_path, exist_ok=True)
        os.makedirs(llm_path, exist_ok=True)
        
        # 设置各个模型的本地路径
        text_encoder_local = os.path.join(llm_path, "gemma-2-2b")
        vae_local = os.path.join(llm_path, "CogVideoX-2b")
        
        # 如果本地不存在则下载模型
        if not os.path.exists(os.path.join(lumina_path, "model_args.pth")):
            snapshot_download(repo_id=video_model_name, local_dir=lumina_path)
            
        if not os.path.exists(text_encoder_local):
            snapshot_download(repo_id=text_encoder_name, local_dir=text_encoder_local)
            
        if not os.path.exists(vae_local):
            snapshot_download(repo_id=vae_model_name, local_dir=vae_local)
        
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[video_model_precision]
        
        # 加载text encoder
        text_encoder = AutoModel.from_pretrained(
            text_encoder_local, 
            torch_dtype=dtype,
            device_map="cuda",
        ).eval()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_local,
        )
        tokenizer.padding_side = "right"
        
        # 加载VAE
        vae = AutoencoderKLCogVideoX.from_pretrained(
            vae_local,
            subfolder="vae",
            torch_dtype=dtype,
        ).cuda()
        
        # 加载主模型
        train_args = torch.load(os.path.join(lumina_path, "model_args.pth"))
        model = models.__dict__[train_args.model](
            in_channels=16,
            qk_norm=train_args.qk_norm,
            cap_feat_dim=text_encoder.config.hidden_size,
            all_patch_size=getattr(train_args, "patch_sizes", (2,)),
            all_f_patch_size=getattr(train_args, "f_patch_sizes", (2,)),
            rope_theta=getattr(train_args, "rope_theta", 10000.0),
            t_scale=getattr(train_args, "t_scale", 1.0),
            motion_scale=getattr(train_args, "motion_scale", 1.0),
        )
        model.eval().to("cuda", dtype=dtype)
        
        # 加载权重
        ckpt_path = os.path.join(lumina_path, "consolidated.00-of-01.safetensors")
        if os.path.exists(ckpt_path):
            ckpt = load_file(ckpt_path)
        else:
            ckpt_path = os.path.join(lumina_path, "consolidated.00-of-01.pth")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
        new_ckpt = {key.replace("_orig_mod.", ""): val for key, val in ckpt.items()}
        model.load_state_dict(new_ckpt, strict=True)
        model.my_compile()
        
        return (model, vae, tokenizer, text_encoder)
