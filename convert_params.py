import paddle
import torch

ckpt_path='/paddle/MiniGPT-4/prerained_minigpt4_7b.pth'
src = torch.load(ckpt_path)['model']
dst = {}
for k, v in src.items():
    # if 'roi_heads.box_' in v and len(src[v].shape)>1:
    #     dst[k] = src[v].transpose(0,1).cpu().numpy().astype('float32')
    # else:
    if src[k].dtype==torch.float32:
        dst[k] = src[k].cpu().numpy().astype('float32')
    elif src[k].dtype==torch.int64:
        dst[k] = src[k].cpu().numpy().astype('int64')
    else:
        print(1)

paddle.save(dst, 'prerained_minigpt4_7b.pdparams')