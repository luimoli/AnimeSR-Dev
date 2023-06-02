"""inference AnimeSR on frames"""
import argparse
import cv2
import glob
import numpy as np
import os
import psutil
import queue
import threading
import time
import torch
from os import path as osp
from tqdm import tqdm

import os,sys
sys.path.append(os.getcwd())

from animesr.utils.inference_base import get_base_argument_parser, get_inference_model
from animesr.utils.video_util import frames2video
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img


def read_img(path, require_mod_crop=True, mod_scale=4, input_rescaling_factor=1.0):
    """ read an image tensor from a given path
    Args:
        path: image path
        require_mod_crop: mod crop or not. since the arch is multi-scale, so mod crop is needed by default
        mod_scale: scale factor for mod_crop


    Returns:
        torch.Tensor: size(1, c, h, w)
    """
    img = cv2.imread(path)
    img = img.astype(np.float32) / 255.

    if input_rescaling_factor != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w * input_rescaling_factor), int(h * input_rescaling_factor)), interpolation=cv2.INTER_LANCZOS4)

    if require_mod_crop:
        img = mod_crop(img, mod_scale)

    img = img2tensor(img, bgr2rgb=True, float32=True)
    return img.unsqueeze(0)

def read_img_crop(path, require_mod_crop=True, mod_scale=4, input_rescaling_factor=1.0, rows=2, cols=2, offset=52):
    """ read an image tensor from a given path
    Args:
        path: image path
        require_mod_crop: mod crop or not. since the arch is multi-scale, so mod crop is needed by default
        mod_scale: scale factor for mod_crop


    Returns:
        torch.Tensor: size(1, c, h, w)
    """
    img = cv2.imread(path)
    img = img.astype(np.float32) / 255.

    if input_rescaling_factor != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w * input_rescaling_factor), int(h * input_rescaling_factor)), interpolation=cv2.INTER_LANCZOS4)

    if require_mod_crop:
        img = mod_crop(img, mod_scale)

    img_patches = crop_image(img, rows, cols, offset)
    img_patches = [img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0) for img in img_patches]
    return img_patches


def crop_image(img, rows, cols, offset):
    height, width, c = img.shape
    assert height % rows == 0 and width % cols == 0
    crop_h, crop_w = height//rows, width//cols

    # judge crop_size
    if (crop_h + offset) % 4 :
        print('------------------------------------------------------')
        print('this offset is wrong!')
        print('new offset:', offset + (crop_h + offset) % 4)
        print('------------------------------------------------------')

        raise ValueError(offset)


    cropped_images = []
    minr, maxr = 0, rows-1
    minc, maxc = 0, cols-1
    for i in range(rows):
        for j in range(cols):
            offset_hw = [offset for i in range(4)]
            if i == minr: offset_hw[0] = 0
            elif i == maxr: offset_hw[1] = 0
            if j == minc: offset_hw[2] = 0
            elif j == maxc: offset_hw[3] = 0
            h1 = i*crop_h - offset_hw[0]
            h2 = (i+1)*crop_h + offset_hw[1]
            w1 = j*crop_w - offset_hw[2]
            w2 = (j+1)*crop_w + offset_hw[3]
            img_patch = img[h1:h2, w1:w2, :].copy()
            cropped_images.append(img_patch)
    return cropped_images

def merge_image(img_patches, rows=2, cols=2, offset=52, scale_factor=4):
    """
    img_patches: tensor imgs: size: [x,c,h,w]
    """
    img0 = img_patches[0]
    c, h, w = img0.size()[-3:]
    h_large = (h // scale_factor - offset) * rows * scale_factor
    w_large = (w // scale_factor - offset) * cols * scale_factor

    img_result = np.zeros((h_large, w_large, c))
    crop_h, crop_w = h_large//rows, w_large//cols

    # minr, maxr = 0, rows-1
    # minc, maxc = 0, cols-1
    tag = 0
    for i in range(rows):
        for j in range(cols):
            offset_hw = [offset*scale_factor for i in range(4)]
            if i == 0: offset_hw[0] = 0
            if j == 0: offset_hw[1] = 0
            
            h1 = offset_hw[0]
            h2 = crop_h + offset_hw[0]
            w1 = offset_hw[1]
            w2 = crop_w + offset_hw[1]

            img_patch = tensor2img(img_patches[tag].cpu().clone().squeeze(0))
            img_result[i*crop_h: (i+1)*crop_h, j*crop_w: (j+1)*crop_w, :]=img_patch[h1:h2, w1:w2, :].copy()
            tag+=1
    return img_result




class IOConsumer(threading.Thread):
    """Since IO time can take up a significant portion of the total inference time,
    so we use multi thread to write frames individually.
    """

    def __init__(self, args: argparse.Namespace, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.args = args

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            imgname = msg['imgname']
            # out_img = tensor2img(output.squeeze(0))
            out_img = output
            if self.args.outscale != self.args.netscale:
                h, w = out_img.shape[:2]
                out_img = cv2.resize(
                    out_img, (int(
                        w * self.args.outscale / self.args.netscale), int(h * self.args.outscale / self.args.netscale)),
                    interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(imgname, out_img)

        print(f'IO for worker {self.qid} is done.')


@torch.no_grad()
def main():
    """Inference demo for AnimeSR.
    It mainly for restoring anime frames.
    """
    parser = get_base_argument_parser()
    parser.add_argument('--input_rescaling_factor', type=float, default=1.0)
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=1,
        help='save 1 frame for every $sample_interval frames. this will be useful for calculating the metrics')
    parser.add_argument('--save_video_too', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)

    # prepare output dir
    frame_output = osp.join(args.output, args.expname, 'frames')
    os.makedirs(frame_output, exist_ok=True)

    # the input format can be:
    # 1. clip folder which contains frames
    # or 2. a folder which contains several clips
    first_level_dir = len(glob.glob(osp.join(args.input, '*.png'))) > 0
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if first_level_dir:
        videos_name = [osp.basename(args.input)]
        args.input = osp.dirname(args.input)
    else:
        videos_name = sorted(os.listdir(args.input))

    pbar1 = tqdm(total=len(videos_name), unit='video', desc='inference')

    que = queue.Queue()
    consumers = [IOConsumer(args, que, f'IO_{i}') for i in range(args.num_io_consumer)]
    for consumer in consumers:
        consumer.start()

    for video_name in videos_name:
        video_folder_path = osp.join(args.input, video_name)
        imgs_list = sorted(glob.glob(osp.join(video_folder_path, '*')))
        num_imgs = len(imgs_list)
        os.makedirs(osp.join(frame_output, video_name), exist_ok=True)



        # prepare
        prev = [img.to(device) for img in read_img_crop(
            imgs_list[0],
            require_mod_crop=True,
            mod_scale=args.mod_scale,
            input_rescaling_factor=args.input_rescaling_factor)]
        cur = prev
        nxt = [img.to(device) for img in read_img_crop(
            imgs_list[min(1, num_imgs - 1)],
            require_mod_crop=True,
            mod_scale=args.mod_scale,
            input_rescaling_factor=args.input_rescaling_factor)]
        # c, h, w = prev.size()[-3:]
        # state = prev.new_zeros(1, 64, h, w)
        # out = prev.new_zeros(1, c, h * args.netscale, w * args.netscale)
        state = [prev[i].new_zeros(1, 64, prev[i].size()[-2], prev[i].size()[-1]) for i in range(len(prev))]
        out = [prev[i].new_zeros(1, prev[i].size()[-3], prev[i].size()[-2]* args.netscale, prev[i].size()[-1]* args.netscale) for i in range(len(prev))]


        pbar2 = tqdm(total=num_imgs, unit='frame', desc='inference')
        tot_model_time = 0
        cnt_model_time = 0
        for idx in range(num_imgs):
            torch.cuda.synchronize()
            start = time.time()
            img_name = osp.splitext(osp.basename(imgs_list[idx]))[0]

            # out, state = model.cell(torch.cat((prev, cur, nxt), dim=1), out, state)
            out_list, state_list = [], []
            for i in range(len(prev)):
                out_tmp, state_tmp = model.cell(torch.cat((prev[i], cur[i], nxt[i]), dim=1), out[i], state[i])
                out_list.append(out_tmp)
                state_list.append(state_tmp)
            out = out_list
            state = state_list

            torch.cuda.synchronize()
            model_time = time.time() - start
            tot_model_time += model_time
            cnt_model_time += 1

            if (idx + 1) % args.sample_interval == 0:
                # put the output frame to the queue to be consumed
                # que.put({'output': out.cpu().clone(), 'imgname': osp.join(frame_output, video_name, f'{img_name}.png')})
                result = merge_image(out)
                que.put({'output': result, 'imgname': osp.join(frame_output, video_name, f'{img_name}.png')})


            torch.cuda.synchronize()
            start = time.time()
            prev = cur
            cur = nxt
            nxt = [img.to(device) for img in read_img_crop(
                imgs_list[min(idx + 2, num_imgs - 1)],
                require_mod_crop=True,
                mod_scale=args.mod_scale,
                input_rescaling_factor=args.input_rescaling_factor)]
            torch.cuda.synchronize()
            read_time = time.time() - start

            pbar2.update(1)
            pbar2.set_description(f'read_time: {read_time}, model_time: {tot_model_time/cnt_model_time}')

            mem = psutil.virtual_memory()
            # since the speed of producer (model inference) is faster than the consumer (I/O)
            # if there is a risk of OOM, just sleep to let the consumer work
            if mem.percent > 80.0:
                time.sleep(30)

        pbar1.update(1)

    for _ in range(args.num_io_consumer):
        que.put('quit')
    for consumer in consumers:
        consumer.join()

    if not args.save_video_too:
        return

    # convert the frames to videos
    video_output = osp.join(args.output, args.expname, 'videos')
    os.makedirs(video_output, exist_ok=True)
    for video_name in videos_name:
        out_path = osp.join(video_output, f'{video_name}.mp4')
        frames2video(
            osp.join(frame_output, video_name), out_path, fps=24 if args.fps is None else args.fps, suffix='png')


if __name__ == '__main__':
    main()
