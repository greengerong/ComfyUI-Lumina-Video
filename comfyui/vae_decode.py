import torch
import comfy.model_management as mm

from diffusers.video_processor import VideoProcessor

class LuminaVideoVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "显存减少但可能引入拼接痕迹"}),
                "tile_sample_min_height": ("INT", {"default": 240, "min": 16, "max": 2048, "step": 8, "tooltip": "最小切片高度，默认值为高度的一半"}),
                "tile_sample_min_width": ("INT", {"default": 360, "min": 16, "max": 2048, "step": 8, "tooltip": "最小切片宽度，默认值为宽度的一半"}),
                "tile_overlap_factor_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                "tile_overlap_factor_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "自动根据尺寸调整切片大小，默认是自动调整"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "Lumina-Video"

    def decode(self, vae, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width,
               tile_overlap_factor_height, tile_overlap_factor_width, auto_tile_size=True):
        print(f"LuminaVideoVAEDecode[INFO] 开始解码操作. enable_vae_tiling={enable_vae_tiling}")
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        latents = samples["samples"]

        # 尝试启用 slicing 以减少显存占用
        try:
            print("LuminaVideoVAEDecode[DEBUG] 尝试启用 slicing")
            vae.enable_slicing()
            print("LuminaVideoVAEDecode[INFO] slicing 启用成功")
        except Exception as e:
            print(f"LuminaVideoVAEDecode[WARNING] 启用 slicing 失败，错误信息: {e}")

        vae.to(device)
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
            print("LuminaVideoVAEDecode[DEBUG] 禁用 tiling")
            vae.disable_tiling()

        # 将 latent 数据转换到设备与正确的数据类型，并调整维度顺序为 [batch, channels, frames, height, width]
        print("LuminaVideoVAEDecode[DEBUG] 转换 latent 数据到目标设备并调整数据格式")
        latents = latents.to(vae.dtype).to(device)
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = (1 / vae.config.scaling_factor) * latents
        
        try:
            print("LuminaVideoVAEDecode[DEBUG] 尝试清除 fake context parallel cache")
            vae._clear_fake_context_parallel_cache()
            print("LuminaVideoVAEDecode[INFO] 清除 cache 成功")
        except Exception as e:
            pass

        try:
            print("LuminaVideoVAEDecode[INFO] 开始 decode 操作")
            decoded = vae.decode(latents)
            frames = decoded.sample
            print("LuminaVideoVAEDecode[INFO] decode 操作成功")
        except Exception as e:
            print(f"LuminaVideoVAEDecode[ERROR] decode 操作失败，错误信息: {e}")
            mm.soft_empty_cache()
            raise e

        vae.disable_tiling()
        vae.to(offload_device)
        mm.soft_empty_cache()

        print("LuminaVideoVAEDecode[INFO] 开始视频后处理")
        video_processor = VideoProcessor(vae_scale_factor=8)
        video_processor.config.do_resize = False

        video = video_processor.postprocess_video(video=frames, output_type="pt")
        video = video[0].permute(0, 2, 3, 1).cpu().float()
        print("LuminaVideoVAEDecode[INFO] 视频处理完毕，返回视频结果")
  
        return (video,)
