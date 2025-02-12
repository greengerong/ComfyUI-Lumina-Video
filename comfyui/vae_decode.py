import torch
from torchvision.transforms.functional import to_pil_image

class LuminaVideoVAEDecode:
    """Lumina Video VAE 解码节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tile_size": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 4096, 
                    "step": 32
                }),
                "overlap": ("INT", {
                    "default": 64, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 32
                }),
                "temporal_size": ("INT", {
                    "default": 64, 
                    "min": 8, 
                    "max": 4096, 
                    "step": 4,
                    "tooltip": "Amount of frames to decode at a time."
                }),
                "temporal_overlap": ("INT", {
                    "default": 8, 
                    "min": 4, 
                    "max": 4096, 
                    "step": 4,
                    "tooltip": "Amount of frames to overlap."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE")
    FUNCTION = "decode"
    CATEGORY = "Lumina-Video"

    def decode(self, vae, samples, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        """使用分块解码方式处理 VAE 解码"""
        print(f"start vae.decode with tile_size={tile_size}, overlap={overlap}, temporal_size={temporal_size}, temporal_overlap={temporal_overlap}")
        
        # 调整重叠大小
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
            
        # 获取时间维度压缩率
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None
            
        # 获取空间维度压缩率
        compression = vae.spacial_compression_decode()
        
        # 使用分块解码
        images = vae.decode_tiled(
            samples["samples"],  # 从 LATENT 字典中获取样本
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap
        )
        
        print(f"vae.decode finished")
        
        # 处理批次维度
        if len(images.shape) == 5:  # 合并批次维度
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
        # 标准化处理
        images = (images + 1.0) / 2.0
        images.clamp_(0.0, 1.0)
        
        return (images,) 
