import torch
import comfy.model_management as mm
from diffusers.video_processor import VideoProcessor

class LuminaVideoVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),               
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Drastically reduces memory use but may introduce seams"
                }),
                "tile_sample_min_height": ("INT", {
                    "default": 240, 
                    "min": 16, 
                    "max": 2048, 
                    "step": 8, 
                    "tooltip": "Minimum tile height, default is half the height"
                }),
                "tile_sample_min_width": ("INT", {
                    "default": 360, 
                    "min": 16, 
                    "max": 2048, 
                    "step": 8, 
                    "tooltip": "Minimum tile width, default is half the width"
                }),
                "tile_overlap_factor_height": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001
                }),
                "tile_overlap_factor_width": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001
                }),
                "auto_tile_size": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Auto size based on height and width, default is half the size"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "Lumina-Video"

    def decode(self, samples, vae, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, 
               tile_overlap_factor_height, tile_overlap_factor_width, auto_tile_size=True):
        """使用 VAE 解码 latent samples"""
        try:
            mm.soft_empty_cache()
            print(f"LuminaVideoVAEDecode[INFO] Starting VAE decode process. enable_vae_tiling= {enable_vae_tiling}, auto_tile_size= {auto_tile_size}")
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            
            # 获取 latents
            latents = samples["samples"]
            print(f"LuminaVideoVAEDecode[DEBUG] Original latents shape: {latents.shape}, channels: {latents.shape[1]}, dtype: {latents.dtype}")
            
            # 启用 slicing（如果支持）
            try:
                vae.enable_slicing()
            except:
                print(f"LuminaVideoVAEDecode[WARNING] VAE slicing not supported")
                pass

            # 移动 VAE 到目标设备
            vae.to(device)
            
            # 配置 tiling
            print(f"LuminaVideoVAEDecode[INFO] Enabling VAE tiling. enable_vae_tiling= {enable_vae_tiling}, auto_tile_size= {auto_tile_size}")
            if enable_vae_tiling:
                if auto_tile_size:
                    vae.enable_tiling()
                else:
                    vae.enable_tiling(
                        tile_sample_min_height=tile_sample_min_height,
                        tile_sample_min_width=tile_sample_min_width,
                        tile_overlap_factor_height=tile_overlap_factor_height,
                        tile_overlap_factor_width=tile_overlap_factor_width,
                    )
            else:
                print(f"LuminaVideoVAEDecode[INFO] VAE tiling disabled")
                vae.disable_tiling()

            # 准备 latents
            print(f"LuminaVideoVAEDecode[INFO] Preparing latents: {latents.shape}")
            latents = latents.to(vae.dtype).to(device)
            
            # 应用缩放因子
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
            latents = 1 / scaling_factor * latents

            # 清理缓存（如果支持）
            try:
                vae._clear_fake_context_parallel_cache()
            except:
                pass

            # 解码
            print(f"LuminaVideoVAEDecode[INFO] Starting VAE decode")
            try:
                frames = vae.decode(latents).sample
            except Exception as e:
                print(f"LuminaVideoVAEDecode[WARNING] First decode attempt failed: {e}, retrying with tiling")
                mm.soft_empty_cache()
                vae.enable_tiling()
                raise e

            print(f"LuminaVideoVAEDecode[DEBUG] Decoded frames shape: {frames.shape}")

            # 清理
            vae.disable_tiling()
            vae.to(offload_device)
            mm.soft_empty_cache()

            # 后处理
            
            print(f"LuminaVideoVAEDecode[INFO] Post-processing frames") 
            video_processor = VideoProcessor(vae_scale_factor=8)
            video_processor.config.do_resize = False

            video = video_processor.postprocess_video(video=frames, output_type="pt")
            print(f"LuminaVideoVAEDecode[DEBUG] Post-processed video shape: {video.shape}")
            video = video[0].permute(0, 2, 3, 1).cpu().float()
            print(f"LuminaVideoVAEDecode[DEBUG] Final video shape: {video.shape}")

            return (video,)
            
        except Exception as e:
            print(f"LuminaVideoVAEDecode[ERROR] VAE decode failed: {e}")
            mm.unload_all_models()
            mm.soft_empty_cache()
            raise e
        finally:
             pass