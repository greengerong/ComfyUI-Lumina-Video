**!!!!还是开发中，开发工作仍未完成！！！**

# ComfyUI-Lumina-Video

这是一个用于ComfyUI的基于 Lumina Video 模型的视频生成插件实现。

## 功能特点

- 支持文本到视频的生成
- 支持自定义分辨率和帧率
- 支持自定义系统提示词和负面提示词
- 支持多种精度模式 (bf16/fp16/fp32)
- 自动下载并管理所需模型

## 安装方法

1. 确保已安装 ComfyUI
2. 克隆本仓库到 ComfyUI 的 custom_nodes 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/greengerong/ComfyUI-Lumina-Video.git

pip install -r requirements.txt
```

## 使用方法

插件提供了以下节点：

### LuminaVideoModelLoader
加载所需的模型文件。

参数：
- video_model_name: 视频模型名称
- video_model_precision: 模型精度 (bf16/fp16/fp32)
- text_encoder_name: 文本编码器名称
- vae_model_name: VAE 模型名称

返回：
- model: 主模型
- vae: VAE模型
- tokenizer: 分词器
- text_encoder: 文本编码器

### LuminaVideoSampler
生成视频内容。

参数：
- model: 主模型
- vae: VAE模型
- tokenizer: 分词器
- text_encoder: 文本编码器
- prompt: 生成提示词
- negative_prompt: 负面提示词
- system_prompt: 系统提示词
- resolution_width: 视频宽度
- resolution_height: 视频高度
- fps: 帧率
- frames: 帧数
- seed: 随机种子
- sample_config: 采样配置

返回：
- video_path: 生成的视频文件路径
- images: 生成的图像序列

## 工作流示例

1. 添加 LuminaVideoModelLoader 节点
2. 添加 LuminaVideoSampler 节点
3. 连接 ModelLoader 的输出到 Sampler 的对应输入
4. 设置生成参数
5. 运行工作流

## 注意事项
- 首次运行时会自动下载所需模型，请确保网络连接正常
- 建议使用 GPU 运行，需要较大的显存
- 生成的视频文件将保存在 ComfyUI 的输出目录中

## 致谢

- [Lumina Video](https://github.com/Alpha-VLLM/Lumina-Video)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
