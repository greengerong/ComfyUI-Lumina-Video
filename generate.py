import argparse
from datetime import datetime
import json
import logging
import os

from diffusers.models import AutoencoderKLCogVideoX
import imageio
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoTokenizer

from configs.sample import CANDIDATE_SAMPLE_CONFIGS
import models
from transport import Sampler, create_transport
from utils.parallel import find_free_port, set_sequence_parallel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an assistant designed to generate high-quality videos with the highest degree of image-text alignment based on user prompts. <Prompt Start> "  # noqa
DEFAULT_PROMPT = "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."  # noqa


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer):
    captions = prompt_batch

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
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


class VideoSampler:
    def __init__(self, args):
        # ************ prepare everything **********
        train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
        logger.info("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

        logger.info(f"Creating lm: Gemma-2-2B")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

        text_encoder = AutoModel.from_pretrained(
            "google/gemma-2-2b", torch_dtype=dtype, device_map="cuda", token=args.hf_token
        ).eval()
        cap_feat_dim = text_encoder.config.hidden_size

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=args.hf_token)
        tokenizer.padding_side = "right"

        vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=dtype).cuda()

        logger.info(f"Creating DiT: {train_args.model}")
        # latent_size = train_args.image_size // 8
        model = models.__dict__[train_args.model](
            in_channels=16,
            qk_norm=train_args.qk_norm,
            cap_feat_dim=cap_feat_dim,
            all_patch_size=getattr(train_args, "patch_sizes", (2,)),
            all_f_patch_size=getattr(train_args, "f_patch_sizes", (2,)),
            rope_theta=getattr(train_args, "rope_theta", 10000.0),
            t_scale=getattr(train_args, "t_scale", 1.0),
            motion_scale=getattr(train_args, "motion_scale", 1.0),
        )
        model.eval().to("cuda", dtype=dtype)

        ckpt_path = os.path.join(
            args.ckpt,
            f"consolidated.{0:02d}-of-{1:02d}.safetensors",
        )
        if os.path.exists(ckpt_path):
            ckpt = load_file(ckpt_path)
        else:
            ckpt_path = os.path.join(
                args.ckpt,
                f"consolidated.{0:02d}-of-{1:02d}.pth",
            )
            assert os.path.exists(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        new_ckpt = {}
        for key, val in ckpt.items():
            new_ckpt[key.replace("_orig_mod.", "")] = val  # remove additional segments involved by compile
        del ckpt
        model.load_state_dict(new_ckpt, strict=True)
        del new_ckpt

        model.my_compile()

        self.dtype = dtype
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def sample_one_config(self, resolution, fps, frames, prompt, neg_prompt, sample_config, out_dir, seed=None):
        # begin sampler
        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
        )
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method="euler",
            num_steps=sample_config["step"],
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=sample_config["ts"],
        )
        # end sampler

        w, h = resolution.split("x")
        w, h = int(w), int(h)
        latent_w, latent_h = w // 8, h // 8
        latent_f = frames // 4

        if seed is not None:
            torch.random.manual_seed(int(seed))

        z = torch.randn([1, 16, latent_f, latent_h, latent_w], device="cuda", dtype=self.dtype)

        real_pos_prompt = SYSTEM_PROMPT + prompt
        real_neg_prompt = SYSTEM_PROMPT + neg_prompt

        cap_feats, cap_mask = encode_prompt([real_pos_prompt, real_neg_prompt], self.text_encoder, self.tokenizer)

        cap_mask = cap_mask.to(cap_feats.device)

        model_kwargs = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            motion_score=[sample_config["motion"], sample_config["negMotion"]],
            patch_comb=sample_config["P"],
            cfg_scale=[sample_config["cfg"]],
            renorm_cfg=sample_config["renorm"],
            print_info=True,
        )
        z = z.repeat(2, 1, 1, 1, 1)

        logger.info("> start sample")
        sample = sample_fn(z, self.model.forward_with_multi_cfg, **model_kwargs)[-1]

        factor = 1.15258426
        sample = sample[:1]
        sample = self.vae.decode((sample / factor).to(self.dtype)).sample.float()[0]
        self.vae._clear_fake_context_parallel_cache()

        sample = (sample + 1.0) / 2.0
        sample.clamp_(0.0, 1.0)

        generated_images = [to_pil_image(_) for _ in sample.unbind(dim=1)]

        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(out_dir, f"{timestr}-{seed}.mp4")
        with imageio.get_writer(out_path, fps=fps) as writer:
            for img in generated_images:
                img_np = np.array(img)
                writer.append_data(img_np)

        info_path = os.path.join(out_dir, f"{timestr}-{seed}.json")
        with open(info_path, "w") as f:
            json.dump(
                {
                    "resolution": resolution,
                    "fps": fps,
                    "frames": frames,
                    "prompt": prompt,
                    "neg_prompt": neg_prompt,
                    "config": sample_config,
                    "seed": seed,
                },
                f,
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")

    parser.add_argument("--resolution", default="672x384", type=str)
    parser.add_argument("--fps", default=16, type=int, choices=[16, 24])
    parser.add_argument("--frames", default=64, type=int)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument(
        "--sample_config", type=str, default="f16F64R512", choices=list(CANDIDATE_SAMPLE_CONFIGS.keys())
    )

    args = parser.parse_known_args()[0]

    os.environ["MASTER_PORT"] = str(find_free_port(10000, 11000))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)

    dist.init_process_group("nccl")
    set_sequence_parallel(1)

    out_dir = f"generated/"
    os.makedirs(out_dir, exist_ok=True)

    sample_config = CANDIDATE_SAMPLE_CONFIGS[args.sample_config]
    sampler = VideoSampler(args)
    sampler.sample_one_config(
        args.resolution, args.fps, args.frames, args.prompt, args.neg_prompt, sample_config, out_dir, seed=1000
    )


if __name__ == "__main__":
    main()
