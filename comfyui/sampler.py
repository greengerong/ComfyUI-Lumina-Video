import os
import torch
import numpy as np
from datetime import datetime
import imageio
from torchvision.transforms.functional import to_pil_image

from ..configs.sample import CANDIDATE_SAMPLE_CONFIGS
from ..transport import Sampler, create_transport

DEFAULT_SYSTEM_PROMPT = "You are an assistant designed to generate high-quality videos with the highest degree of image-text alignment based on user prompts. <Prompt Start> "
DEFAULT_PROMPT = "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."  # noqa
class LuminaVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "tokenizer": ("TOKENIZER",),
                "text_encoder": ("TEXT_ENCODER",),
                "prompt": ("STRING", {
                    "default": DEFAULT_PROMPT, 
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True
                }),
                "system_prompt": ("STRING", {
                    "default": DEFAULT_SYSTEM_PROMPT, 
                    "multiline": True
                }),
                "resolution_width": ("INT", {
                    "default": 1248, 
                    "step": 8
                }),
                "resolution_height": ("INT", {
                    "default": 704, 
                    "step": 8
                }),
                "fps": ("INT", {
                    "default": 24, 
                    "min": 1, 
                    "max": 60
                }),
                "frames": ("INT", {
                    "default": 96, 
                }),
                "seed": ("INT", {
                    "default": 888888, 
                    "min": 0,
                }),
                "sample_config": (list(CANDIDATE_SAMPLE_CONFIGS.keys()), {
                    "default": "f24F96R960"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE",)  # 返回视频路径和图像列表
    RETURN_NAMES = ("video_path", "images",)
    FUNCTION = "generate"
    CATEGORY = "Lumina-Video"

    def generate(self, model, vae, tokenizer, text_encoder, prompt, negative_prompt, system_prompt, 
                resolution_width, resolution_height, fps, frames, seed, sample_config):
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        
        sample_config_dict = CANDIDATE_SAMPLE_CONFIGS[sample_config]
        
        # 创建transport和 sampler
        transport = create_transport("Linear", "velocity", None, None, None)
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method="euler",
            num_steps=sample_config_dict["step"],
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=sample_config_dict["ts"],
        )
        
        # 设置分辨率
        w, h = resolution_width, resolution_height
        latent_w, latent_h = w // 8, h // 8
        latent_f = frames // 4
        
        if seed is not None:
            torch.random.manual_seed(seed)
            
        z = torch.randn([1, 16, latent_f, latent_h, latent_w], device="cuda", dtype=next(model.parameters()).dtype)
        
        # 处理提示词
        real_pos_prompt = system_prompt + prompt
        real_neg_prompt = system_prompt + negative_prompt
        
        cap_feats, cap_mask = encode_prompt([real_pos_prompt, real_neg_prompt], text_encoder, tokenizer)
        cap_mask = cap_mask.to(cap_feats.device)
        
        model_kwargs = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            motion_score=[sample_config_dict["motion"], sample_config_dict["negMotion"]],
            patch_comb=sample_config_dict["P"],
            cfg_scale=[sample_config_dict["cfg"]],
            renorm_cfg=sample_config_dict["renorm"],
            print_info=True,
        )
        z = z.repeat(2, 1, 1, 1, 1)
        
        # 生成采样
        sample = sample_fn(z, model.forward_with_multi_cfg, **model_kwargs)[-1]
        
        factor = 1.15258426
        sample = sample[:1]
        sample = vae.decode((sample / factor).to(next(model.parameters()).dtype)).sample.float()[0]
        vae._clear_fake_context_parallel_cache()
        
        sample = (sample + 1.0) / 2.0
        sample.clamp_(0.0, 1.0)
        
        generated_images = [to_pil_image(_) for _ in sample.unbind(dim=1)]
        
        # 保存视频
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(out_dir, f"{timestr}-{seed}.mp4")
        with imageio.get_writer(out_path, fps=fps) as writer:
            for img in generated_images:
                writer.append_data(np.array(img))
        
        return (out_path, generated_images)

def encode_prompt(prompt_batch, text_encoder, tokenizer):
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks
