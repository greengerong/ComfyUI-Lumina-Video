import torch
import comfy
import numpy as np
from .model_loader import load_video_model, load_text_model
from .utils import save_video_frames

SYSTEM_PROMPT = "You are an assistant designed to generate high-quality videos..."

class LuminaVideoGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": (["Alpha-VLLM/Lumina-Video-f24R960"], {"default": "Alpha-VLLM/Lumina-Video-f24R960"}),
                "text_encoder": (["google/gemma-2-2b"], {"default": "google/gemma-2-2b"}),
                "resolution": ("STRING", {"default": "1248x704"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "frames": ("INT", {"default": 96, "min": 16, "max": 256}),
                "prompt": ("STRING", {"multiline": True, "default": "A large orange octopus..."}),
                "negative_prompt": ("STRING", {"default": ""}),
                "sample_config": (["f24F96R960", "f24F96R960-MultiScale"], {"default": "f24F96R960"}),
                "seed": ("INT", {"default": 0}),
            }
        }

    CATEGORY = "Lumina/Video"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    
    def __init__(self):
        self.video_model = None
        self.text_encoder = None
        self.tokenizer = None
        
    def load_models(self, video_model_name, text_encoder_name):
        # 加载视频生成模型
        self.video_model = load_video_model(video_model_name)
        
        # 加载文本编码器
        self.text_encoder, self.tokenizer = load_text_model(text_encoder_name)
    
    def generate(self, video_model, text_encoder, resolution, fps, frames, prompt, 
                negative_prompt, sample_config, seed):
        if self.video_model is None:
            self.load_models(video_model, text_encoder)
            
        # 准备潜在空间参数
        w, h = map(int, resolution.split("x"))
        latent_w, latent_h = w // 8, h // 8
        latent_f = frames // 4
        
        # 生成随机噪声
        torch.manual_seed(seed)
        z = torch.randn([1, 16, latent_f, latent_h, latent_w], device="cuda", dtype=torch.bfloat16)
        
        # 编码提示词
        pos_feats = self.encode_prompt(prompt)
        neg_feats = self.encode_prompt(negative_prompt)
        cap_feats = torch.cat([pos_feats, neg_feats])
        
        # 执行采样
        sample = self.video_model.sample(
            z, 
            cap_feats=cap_feats,
            cfg_scale=7.0,
            num_steps=96 if "MultiScale" in sample_config else 50
        )
        
        # 解码并保存视频
        output_path = save_video_frames(sample, fps, seed)
        return (output_path,)

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            [SYSTEM_PROMPT + prompt],
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        return self.text_encoder(
            input_ids=text_inputs.input_ids.cuda(),
            attention_mask=text_inputs.attention_mask.cuda(),
        ).hidden_states[-2]
