import argparse
import cv2
import glob
import numpy as np
import os
import torch

from drct.archs.DRCT_arch import *
#from drct.data import *
#from drct.models import *

from torchvision.transforms import functional as TF

def apply_tta(img):
    return [
        (lambda x: x, 'none'),
        (TF.hflip, 'hflip'),
        (TF.vflip, 'vflip'),
        (lambda x: TF.hflip(TF.vflip(x)), 'hvflip')
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        "/work/u1657859/DRCT/experiments/train_DRCT-L_SRx4_finetune_from_ImageNet_pretrain/models/DRCT-L.pth"  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/DRCT-L', help='output folder')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4')
    #parser.add_argument('--window_size', type=int, default=16, help='16')
    
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model (DRCT-L)
    model = DRCT(upscale=4, in_chans=3,  img_size= 64, window_size= 16, compress_ratio= 3,squeeze_factor= 30,
                        conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim= 180, num_heads= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], gc= 32,
                        mlp_ratio= 2, upsampler= 'pixelshuffle', resi_connection= '1conv')
    
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)
    
    print(model)
    
    window_size = 16
    
    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        
        #img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        #print(img.shape)
        # inference
        try:
            with torch.no_grad():
                #output = model(img)

                outputs = []
                for transform, name in apply_tta(img[0]): 
                    tta_img = transform(img[0]) 
                    tta_img = tta_img.unsqueeze(0).to(device)

                    _, _, h_old, w_old = tta_img.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    tta_img = torch.cat([tta_img, torch.flip(tta_img, [2])], 2)[:, :, :h_old + h_pad, :]
                    tta_img = torch.cat([tta_img, torch.flip(tta_img, [3])], 3)[:, :, :, :w_old + w_pad]
                    output = test(tta_img, model, args, window_size, idx)
                    output = output[..., :h_old * args.scale, :w_old * args.scale]
                    # inverse transform
                    if name == 'hflip':
                        output = torch.flip(output, [3])
                    elif name == 'vflip':
                        output = torch.flip(output, [2])
                    elif name == 'hvflip':
                        output = torch.flip(output, [2, 3])
                    outputs.append(output)

                
                output = torch.stack(outputs).mean(dim=0) 

        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_DRCT-L_X4.png'), output)



def test(img_lq, model, args, window_size, idx):
    if args.tile is None:
        # test the image as a whole
        # output, feature_maps = model(img_lq)
        output, _ = model(img_lq)

        # import matplotlib.pyplot as plt
        # for i, feat in enumerate(feature_maps):
        #     fmap = feat[0].mean(0).cpu().numpy()  # 平均所有 channels，shape: [H, W]
        
        #     plt.figure(figsize=(4, 4))            # 每張開新圖
        #     plt.imshow(fmap, cmap='viridis')
        #     plt.colorbar()
        #     plt.title(f"RDG {i+1} Feature Map (Mean over C)")
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig(f"./visual_feat/{idx}_feature_rdg{i+1}.png")
        #     plt.close()  # 避免重疊
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output



if __name__ == '__main__':
    main()
