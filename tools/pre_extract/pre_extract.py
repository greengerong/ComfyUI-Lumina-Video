import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])

from argparse import ArgumentParser
import functools
import json
import math
import multiprocessing as mp
import pickle
import traceback

from PIL import Image
from diffusers.models import AutoencoderKLCogVideoX
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data import data_reader
from imgproc import generate_crop_size_list, var_center_crop


class ItemProcessor:
    def __init__(self, target_size, target_frame_rate):

        model_patch_size = 4  # the largest patchification has a spatial down sampling of 4
        patch_size = 8 * model_patch_size  # 8 for vae spatial down sampling

        # print(f"patch size for crop size computation: {patch_size}")
        max_num_patches = round((target_size / patch_size) ** 2)
        # print(f"Limiting number of patches to {max_num_patches}.")
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
        # print("List of crop sizes:")
        # for line in range(0, len(crop_size_list), 6):
        #     print(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[line : line + 6]]))
        image_transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.image_transform = image_transform
        self.target_frame_rate = target_frame_rate
        self.target_size = target_size

        self.printed = False

    def process_item(self, data_item):
        try:
            url = data_item["path"]
            video_data = data_reader.read_general(url)

            # decord has to be imported after torch, bug here: https://github.com/dmlc/decord/issues/293
            import decord

            video_reader = decord.VideoReader(video_data)
            del video_data

            if video_reader.get_avg_fps() < self.target_frame_rate / 2:
                raise ValueError(f"fps {video_reader.get_avg_fps():.2f} too low {url}")

            info = {
                "seconds": len(video_reader) / video_reader.get_avg_fps(),
                "fps": video_reader.get_avg_fps(),
                "width": None,
                "height": None,
            }

            seconds = len(video_reader) / video_reader.get_avg_fps()
            # +0.2 so that, say, 15.85 frames can round to 16 frames
            target_frames = int(seconds * self.target_frame_rate + 0.2)

            target_frames = 1 + ((target_frames - 1) // 4 * 4)  # For CogvideoX VAE, num input frames should be 4n+1

            if target_frames < 9:
                return None, None

            start_time = 0.0

            l_frame_ids = []
            for i in range(target_frames):
                frame_id = round((start_time + i / self.target_frame_rate) * video_reader.get_avg_fps())
                if frame_id >= len(video_reader):
                    frame_id = len(video_reader) - 1

                l_frame_ids.append(frame_id)

            frames = video_reader.get_batch(l_frame_ids).asnumpy()

            del video_reader

            if not self.printed:  # for debug
                # print("video fps: ", video_reader.get_avg_fps())
                # print("video F: ", len(video_reader))
                # print("frame size: ", Image.fromarray(frames_npy[0]).size)
                # print("l_frame_ids: ", l_frame_ids)
                self.printed = True

            frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]

            info["width"], info["height"] = frames[0].width, frames[0].height

            if frames[0].width * frames[0].height < self.target_size**2:
                return None, None

            frames = [self.image_transform(_) for _ in frames]

            return torch.stack(frames, dim=1), info
        except:
            import traceback

            print(traceback.format_exc(), flush=True)
            return None, None


class MyDataset(Dataset):
    def __init__(self, _item_processor: ItemProcessor, _meta_list: list):
        super(MyDataset, self).__init__()
        self.meta_list = _meta_list
        self.item_processor = _item_processor

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        result = (idx, self.item_processor.process_item(self.meta_list[idx]), self.meta_list[idx])
        return result


def writer_main(write_queue, write_barrier, args, end_idx):
    write_barrier.wait()
    while True:
        (
            data_idx,
            new_item,
            record,
            info,
        ) = write_queue.get()

        try:
            if new_item is not None:
                pkl_path = os.path.join(args.pkl_dir, f"{data_idx}.pkl")
                if "s3://" in pkl_path:
                    data_reader.init_ceph_client_if_needed()
                    data_reader.client.put(pkl_path, pickle.dumps(new_item))
                else:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(new_item, f)

                record.update({"pkl": pkl_path, "pkl_shape": str(list(new_item["latent"].shape))})
                if "ori_id" not in record:
                    record["ori_id"] = data_idx
                record.update(info)

                if record is not None:
                    with open(os.path.join(args.record_dir, f"{args.rank}-of-{args.splits}-record.jsonl"), "a") as f:
                        record_str = json.dumps(record) + "\n"
                        f.write(record_str)
            else:  # illegal item
                pass

            with open(os.path.join(args.record_dir, f"{args.rank}-of-{args.splits}-progress.txt"), "w") as f:
                if data_idx == end_idx - 1:
                    f.write("finished")
                else:
                    f.write(f"{data_idx}")

        except Exception as e:
            print(traceback.format_exc(), flush=True)

        write_barrier.wait()


def main():
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
        "--pkl_dir",
        type=str,
    )
    parser.add_argument(
        "--record_dir",
        type=str,
    )
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--target_fps", type=int, default=16)
    args = parser.parse_args()

    args.rank = int(os.environ.get("SLURM_PROCID", 0)) + args.rank_bias
    print(f"rank: {args.rank}")

    splits = args.splits
    rank = args.rank
    if "s3://" not in args.pkl_dir:
        os.makedirs(args.pkl_dir, exist_ok=True)
    assert "s3://" not in args.record_dir
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(os.path.join(args.record_dir, "logs"), exist_ok=True)

    log_file = open(os.path.join(args.record_dir, f"logs/{args.rank}-of-{args.splits}-log.txt"), "a")
    sys.stdout = log_file
    sys.stderr = log_file

    torch.cuda.set_device(args.rank % torch.cuda.device_count())

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.bfloat16
    ).cuda()
    vae.eval()
    vae = torch.compile(vae)

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
        print(f"resume from {start_idx}", flush=True)
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}", flush=True)

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))

    item_processor = ItemProcessor(target_size=args.target_size, target_frame_rate=args.target_fps)
    dataset = MyDataset(item_processor, ori_contents[start_idx:end_idx])
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)

    write_queue = mp.Queue()
    write_barrier = mp.Barrier(2)
    write_process = mp.Process(
        target=writer_main,
        args=(write_queue, write_barrier, args, end_idx),
    )
    write_process.start()

    for data_idx, (x, info), record in dataloader:
        with torch.no_grad():
            try:
                # data_idx = data_idx[0].item()
                data_idx += start_idx
                # x = x[0]

                if data_idx % 10 == 0:
                    print(f"{rank}: {start_idx}-{data_idx}-{end_idx}", flush=True)

                if x is None:
                    print(f"illegal x for item {data_idx}")
                    write_barrier.wait()
                    write_queue.put((data_idx, None, None, None))
                    continue

                x = x.to(device="cuda", dtype=torch.bfloat16)  # C, F, H, W
                # if i == start_idx:
                #     print("x shape: ", x.shape)
                vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215, "cogvideox": 1.15258426}[
                    "cogvideox"
                ]
                vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0, "cogvideox": 0.0}["cogvideox"]

                x = list(torch.split(x, 32, dim=1))
                latent = []

                vae._clear_fake_context_parallel_cache()

                for sub_x in x:
                    latent.append(((vae.encode(sub_x[None]).latent_dist.mode()[0] - vae_shift) * vae_scale).cpu())
                latent = torch.cat(latent, dim=1)

                vae._clear_fake_context_parallel_cache()

                new_item = {"latent": latent.cpu(), "id": data_idx}

                write_barrier.wait()
                write_queue.put((data_idx, new_item, record, info))

            except Exception as e:
                # from traceback import format_exc
                # print(f"item {data_idx} error: \n{ori_contents[data_idx]}")
                # print(format_exc())
                pass

    write_barrier.wait()
    write_process.kill()


if __name__ == "__main__":
    main()
