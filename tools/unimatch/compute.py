import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])

from argparse import ArgumentParser
import json
import math

from PIL import Image
from einops import rearrange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from unimatch_model import UniMatch

from data import data_reader


class ItemProcessor:
    def __init__(self, target_size=(320, 576), timestamps=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)):

        self.target_size = target_size
        self.timestamps = timestamps

        self.printed = False

    def process_item(self, data_item):
        try:
            url = data_item["path"]
            video_data = data_reader.read_general(url)

            # decord has to be imported after torch, bug here: https://github.com/dmlc/decord/issues/293
            import decord

            video_reader = decord.VideoReader(video_data)

            l_frame_ids = []
            for t in self.timestamps:
                frame_id = round(t * video_reader.get_avg_fps())
                if frame_id < len(video_reader):
                    l_frame_ids.append(frame_id)
                else:
                    break
            if len(l_frame_ids) < 2:
                raise ValueError()

            frames_npy = video_reader.get_batch(l_frame_ids).asnumpy()

            if not self.printed:
                # print("video fps: ", video_reader.get_avg_fps())
                # print("video F: ", len(video_reader))
                # print("frame size: ", Image.fromarray(frames_npy[0]).size)
                # print("l_frame_ids: ", l_frame_ids)
                self.printed = True

            frames = [Image.fromarray(frames_npy[i]) for i in range(frames_npy.shape[0])]
            frames = torch.stack([pil_to_tensor(frame) for frame in frames]).float()

            H, W = frames.shape[-2:]
            if H > W:
                frames = rearrange(frames, "N C H W -> N C W H")
            frames = F.interpolate(frames, size=self.target_size, mode="bilinear", align_corners=True)
            return frames

        except:
            import traceback

            print(traceback.format_exc(), flush=True)
            return None


class MyDataset(Dataset):
    def __init__(self, _item_processor: ItemProcessor, _meta_list: list):
        super(MyDataset, self).__init__()
        self.meta_list = _meta_list
        self.item_processor = _item_processor

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        return idx, self.item_processor.process_item(self.meta_list[idx]), self.meta_list[idx]


def main():
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank_bias",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--record_dir",
        type=str,
    )
    args = parser.parse_args()

    args.rank = int(os.environ.get("SLURM_PROCID", 0)) + args.rank_bias
    print(f"rank: {args.rank}")

    torch.cuda.set_device(args.rank % torch.cuda.device_count())

    splits = args.splits
    rank = args.rank

    assert "s3://" not in args.record_dir
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(os.path.join(args.record_dir, "logs"), exist_ok=True)

    log_file = open(os.path.join(args.record_dir, f"logs/{args.rank}-of-{args.splits}-log.txt"), "a")
    sys.stdout = log_file
    sys.stderr = log_file

    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    )
    model.eval()
    ckpt = torch.load("unimatch/ckpts/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")  # noqa
    # noqa download from https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
    model.load_state_dict(ckpt["model"])
    model = model.cuda()

    in_filename: str = args.in_filename
    if in_filename.endswith(".json"):
        with open(in_filename, "r") as f:
            ori_contents = json.load(f)
    elif in_filename.endswith(".jsonl"):
        with open(args.in_filename) as f:
            ori_contents = [json.loads(_) for _ in f.readlines()]
    else:
        raise ValueError(f"Unrecognized in_filename: {in_filename}")

    num = len(ori_contents)

    num_per_rank = math.ceil(num / splits)

    try:
        with open(os.path.join(args.record_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            rank_progress = f.read()
            if "finished" in rank_progress:
                print(f"rank {rank} of {splits} finished", flush=True)
                return
            else:
                start_idx = int(rank_progress) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))

    item_processor = ItemProcessor()
    dataset = MyDataset(item_processor, ori_contents[start_idx:end_idx])
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=1, pin_memory=True)

    for data_idx, images, ori_item in dataloader:
        with torch.no_grad():
            record = None
            try:
                # data_idx = data_idx[0].item()
                data_idx += start_idx
                # x = x[0]
                if images is None:
                    raise ValueError(f"illegal x for item {data_idx}")

                if data_idx % 10 == 0:
                    print(f"{rank}: {start_idx}-{data_idx}-{end_idx}", flush=True)

                images = images.to(device="cuda")  # C, F, H, W

                batch_0 = images[:-1]
                batch_1 = images[1:]

                with torch.no_grad():
                    res = model(
                        batch_0,
                        batch_1,
                        attn_type="swin",
                        attn_splits_list=[2, 8],
                        corr_radius_list=[-1, 4],
                        prop_radius_list=[-1, 1],
                        num_reg_refine=6,
                        task="flow",
                        pred_bidir_flow=False,
                    )
                    flow_map = res["flow_preds"][-1]  # [F-1, 2, H, W]
                    flow_score = flow_map.abs().mean().item()

                record = ori_item
                record.update({"unimatch_flow": flow_score})

            except Exception as e:
                from traceback import format_exc

                print(f"item {data_idx} error: \n{ori_contents[data_idx]}")
                print(format_exc())

        if record is not None:
            with open(os.path.join(args.record_dir, f"{rank}-of-{splits}-record.jsonl"), "a") as f:
                record_str = json.dumps(record) + "\n"
                f.write(record_str)

        with open(os.path.join(args.record_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
            if data_idx == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{data_idx}")


if __name__ == "__main__":
    main()
