import torch
import torch.nn as nn
from sam2.build_sam import build_sam2_video_predictor

class RE_SAM2(nn.Module):
    def __init__(self,config=None):
        super(RE_SAM2, self).__init__()
        self.sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
        self.model_cfg = "sam2_hiera_t_med.yaml"
        self.sam2_model = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.img_size = 256

    def train(self):
        # self.VRP_model.train_mode()
        self.sam2_model.train()

    def eval(self):
        # self.VRP_model.eval()
        self.sam2_model.eval()

    def forward(self,qeury_imgs, support_imgs,support_masks):
        b,s,c,h,w = support_imgs.shape
        b_q,c,h,w = qeury_imgs.shape
        mask = support_masks.view(-1,self.img_size,self.img_size)
        supp_img = support_imgs.view(-1,3,self.img_size,self.img_size)

        idx = b_q//2
        video_data = torch.cat([qeury_imgs[:idx],supp_img,qeury_imgs[idx:]], dim=0)
        inference_state = self.sam2_model.init_state(video_data=video_data)
        self.sam2_model.reset_state(inference_state)

        for i in range(b*s):
            frame_idx, out_obj_ids, out_mask_logits = self.sam2_model.add_new_mask(
                inference_state=inference_state,
                frame_idx=idx + i,
                obj_id=1,
                mask = mask[i],
            )

        video_segments = torch.zeros(b_q+s,h,w ).cuda()
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state,start_frame_idx=idx+s):
            video_segments[out_frame_idx] =out_mask_logits[0]
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state,start_frame_idx=idx-1,reverse=True):
            video_segments[out_frame_idx] =out_mask_logits[0]
        out = torch.cat([video_segments[:idx],video_segments[idx+s:]])

        return out
