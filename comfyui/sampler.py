import os
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
import imageio
from torchvision.transforms.functional import to_pil_image
import folder_paths
import json

from ..configs.sample import CANDIDATE_SAMPLE_CONFIGS
from ..transport import Sampler, create_transport
from ..utils.parallel import find_free_port, set_sequence_parallel

def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method=f"tcp://127.0.0.1:{str(find_free_port(10000, 11000))}",
            world_size=1,
            rank=0
        )

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
        
        # 初始化分布式环境
        init_distributed()
        set_sequence_parallel(1)
        print(f"init_distributed finished")
        
        sample_config_dict = CANDIDATE_SAMPLE_CONFIGS[sample_config]
        print(f"sample_config_dict = {json.dumps(sample_config_dict, indent=2)}")
        
        # 创建transport和 sampler
        transport = create_transport("Linear", "velocity", None, None, None)
        sampler = Sampler(transport)
        print(f"start sampler.sample_ode...")
        sample_fn = sampler.sample_ode(
            sampling_method="euler",
            num_steps=sample_config_dict["step"],
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=sample_config_dict["ts"],
        )
        print(f"sampler.sample_ode finished")
        
        # 设置分辨率
        w, h = resolution_width, resolution_height
        latent_w, latent_h = w // 8, h // 8
        latent_f = frames // 4
        
        if seed is not None:
            torch.random.manual_seed(seed)

        print(f"start torch.randn...")
        z = torch.randn([1, 16, latent_f, latent_h, latent_w], device="cuda", dtype=next(model.parameters()).dtype)
        print(f"torch.randn finished")
        
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
        print(f"start sample_fn...")
        sample = sample_fn(z, model.forward_with_multi_cfg, **model_kwargs)[-1]
        print(f"sample_fn finished")
        
        factor = 1.15258426
        sample = sample[:1]

        print(f"start vae.decode...")
        sample = vae.decode((sample / factor).to(next(model.parameters()).dtype)).sample.float()[0]    
        vae._clear_fake_context_parallel_cache()
        print(f"vae.decode finished")
        
        sample = (sample + 1.0) / 2.0
        sample.clamp_(0.0, 1.0)
        
        generated_images = [to_pil_image(_) for _ in sample.unbind(dim=1)]
        print(f"generated_images generated")
        
        # 保存视频
        out_dir = os.path.join(folder_paths.base_path, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(out_dir, f"{timestr}-{seed}.mp4")
        print(f"start with imageio.get_writer to out_path={out_path}...")

        with imageio.get_writer(out_path, fps=fps) as writer:
            for img in generated_images:
                writer.append_data(np.array(img))

        print(f"video saved to {out_path}")

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
