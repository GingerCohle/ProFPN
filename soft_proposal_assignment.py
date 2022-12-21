import torch
import torch.nn as nn

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
        self.finest_scale = finest_scale
        self.epsilon=1e-4


    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls, scale

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls, scale = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            lvl_list = [0, 1, 2 ,3]
            inds = target_lvls == i
            if inds.any():
                lvl_list.remove(i)
                rois_ = rois[inds, :]
                scale_= scale[inds]
                if i == 0:
                    lv0_weights_1 = (1 - (112 - scale_) / 112).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    roi_feats_t = self.roi_layers[0](feats[0], rois_) + lv0_weights_1 * self.roi_layers[1](feats[1], rois_)
                elif i == 1:
                    lv1_weights_0 = (1 - (scale_-112) / 112).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    lv1_weights_2 = (1 - (224-scale_) / 112).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    roi_feats_t =  lv1_weights_0 * self.roi_layers[0](feats[0], rois_) + \
                                   self.roi_layers[1](feats[1], rois_) + \
                                   lv1_weights_2 * self.roi_layers[2](feats[2], rois_)
                elif i == 2:
                    lv2_weights_1 = (1 - (scale_ - 224) / 224).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    lv2_weights_3 = (1-(448 - scale_) / 224).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    roi_feats_t = lv2_weights_1 * self.roi_layers[1](feats[1], rois_) + \
                                  self.roi_layers[2](feats[2], rois_) + \
                                  lv2_weights_3 * self.roi_layers[3](feats[3], rois_)
                elif i == 3:
                    lv3_weight_2 = (1 -(scale_ -448) / 448)
                    ids_l = scale_ >= 448 * 2
                    lv3_weight_2[ids_l]=0
                    roi_feats_t =  self.roi_layers[3](feats[3], rois_) + lv3_weight_2.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.roi_layers[2](feats[2], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
        return roi_feats
