import os
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from diffusers.models import AutoencoderKLCogVideoX
from ..models import MultiScaleNextDiT_2B_GQA
import folder_paths
import json
import comfy.model_management as mm

class LuminaVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_model_name": (["Alpha-VLLM/Lumina-Video-f24R960"], {
                    "default": "Alpha-VLLM/Lumina-Video-f24R960"
                }),
                "video_model_precision": (["bf16", "fp8", "int4"], {
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
        # 在加载新模型之前先清理已加载的模型与缓存，以释放GPU内存
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"清理显存时出现错误: {e}")
            
        # 设置ComfyUI根目录下的models模型路径
        base_path = folder_paths.base_path
        lumina_path = os.path.join(base_path, "models", "Lumina-Video")
        llm_path = os.path.join(base_path, "models", "LLM")
        cogvideo_path = os.path.join(base_path, "models", "CogVideo")
        
        # 确保目录存在
        os.makedirs(lumina_path, exist_ok=True)
        os.makedirs(llm_path, exist_ok=True)
        os.makedirs(cogvideo_path, exist_ok=True)
        
        # 设置各个模型的本地路径
        video_local = os.path.join(lumina_path, "Lumina-Video-f24R960")
        text_encoder_local = os.path.join(llm_path, "gemma-2-2b")
        vae_local = os.path.join(cogvideo_path, "CogVideoX-2b")
        
        # 如果本地不存在则下载模型
        if not os.path.exists(os.path.join(video_local, "model_args.pth")):
            print(f"Downloading Lumina-Video model from {video_model_name} to {video_local}")
            snapshot_download(repo_id=video_model_name, local_dir=video_local, resume_download=True)
        else:
            print(f"Lumina-Video model already exists in {video_local}")
            
        if not os.path.exists(text_encoder_local):
            print(f"Downloading text encoder model from {text_encoder_name} to {text_encoder_local}")
            snapshot_download(repo_id=text_encoder_name, local_dir=text_encoder_local, resume_download=True)
        else:
            print(f"Text encoder model already exists in {text_encoder_local}") 
            
        if not os.path.exists(vae_local):
            print(f"Downloading VAE model from {vae_model_name} to {vae_local}")
            snapshot_download(repo_id=vae_model_name, local_dir=vae_local, resume_download=True, allow_patterns=["vae/**"])
        else:
            print(f"VAE model already exists in {vae_local}")
        
        # 为video model选择精度:
        # 当选择bf16时直接以bf16加载；若选择fp8或int4，则先以fp32加载，后续在编译中启用量化转换
        if video_model_precision == "bf16":
            dtype_video = torch.bfloat16
        else:
            dtype_video = torch.float32
        
        # 加载text encoder，固定使用bf16加载
        text_encoder = AutoModel.from_pretrained(
            text_encoder_local, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_local,
        )
        tokenizer.padding_side = "right"
        print(f"load tokenizer model finished")
        
        # 加载VAE，固定使用bf16加载
        vae = AutoencoderKLCogVideoX.from_pretrained(
            vae_local,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).cuda()
        print(f"load vae model finished")
        
        # 加载主模型
        train_args = torch.load(os.path.join(video_local, "model_args.pth"))
        print(f"load model arguments: {json.dumps(train_args.__dict__, indent=2)}")

        print(f"Creating DiT: {train_args.model}")
        if train_args.model == "MultiScaleNextDiT_2B_GQA":
            model = MultiScaleNextDiT_2B_GQA(
                in_channels=16,
                qk_norm=train_args.qk_norm,
                cap_feat_dim=text_encoder.config.hidden_size,
                all_patch_size=getattr(train_args, "patch_sizes", (2,)),
                all_f_patch_size=getattr(train_args, "f_patch_sizes", (2,)),
                rope_theta=getattr(train_args, "rope_theta", 10000.0),
                t_scale=getattr(train_args, "t_scale", 1.0),
                motion_scale=getattr(train_args, "motion_scale", 1.0),
            )
        else:
            raise ValueError(f"Unknown model type: {train_args.model}")
        
        model.eval().to("cuda", dtype=dtype_video)
        
        # 加载权重
        ckpt_path = os.path.join(video_local, "consolidated.00-of-01.safetensors")
        if os.path.exists(ckpt_path):
            ckpt = load_file(ckpt_path)
        else:
            ckpt_path = os.path.join(video_local, "consolidated.00-of-01.pth")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
        new_ckpt = {key.replace("_orig_mod.", ""): val for key, val in ckpt.items()}
        model.load_state_dict(new_ckpt, strict=True)
        
        # 如果选择更低精度的加载(fp8或int4)，在编译前传递量化模式给模型
        if video_model_precision in ("fp8", "int4"):
            model.quantization_mode = video_model_precision
        
        model.my_compile()
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"Error in unload_all_models() and soft_empty_cache(): {e}")

        return (model, vae, tokenizer, text_encoder)
