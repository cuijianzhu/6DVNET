from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .my_roi_heads import MyRoIHeads
from torch import nn
import torch
from torch.nn import functional as F


class PoseRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(PoseRCNNPredictor, self).__init__()
        self.pose_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        pose_regs = self.pose_pred(x)
        return pose_regs


class MLPFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_bn = nn.BatchNorm1d(100)

        nn.init.constant_(self.fc1_bn.weight, 1.0)
        nn.init.constant_(self.fc2_bn.weight, 1.0)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1_bn(self.fc1(x.view(batch_size, -1))), inplace=True)
        x = F.relu(self.fc2_bn(self.fc2(x)), inplace=True)
        return x


class MLPCONCATPredictor(nn.Module):

    def __init__(self, representation_size):
        super().__init__()
        self.car_rot_linear = nn.Linear(representation_size, 100)
        self.car_rot_linear_bn = nn.BatchNorm1d(100)
        self.trans_pred = nn.Linear(100+100, 3)    # concatenate

        nn.init.constant_(self.car_rot_linear_bn.weight, 1.0)

        nn.init.normal_(self.car_rot_linear.weight, std=0.01)
        nn.init.normal_(self.trans_pred.weight, std=0.1)
        for l in [self.car_rot_linear, self.trans_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x_mlp, x_car_rot):
        if x_mlp.dim() == 4:
            x_mlp = x_mlp.squeeze(3).squeeze(2)

        batch_size = x_mlp.size(0)
        x_car_rot = F.relu(self.car_rot_linear_bn(self.car_rot_linear(x_car_rot.view(batch_size, -1))), inplace=True)
        x_merge = F.relu(torch.cat((x_mlp, x_car_rot), dim=1))

        trans_pred = self.trans_pred(x_merge)
        return trans_pred


class PoseRCNN(FasterRCNN):

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Pose parameters
                 pose_roi_pool=None, pose_head=None, pose_predictor=None,
                 # Translation
                 translation_head=None, translation_predictor=None):

        assert isinstance(pose_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if pose_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels
        # Pose
        if pose_roi_pool is None:
            pose_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if pose_head is None:
            resolution = pose_roi_pool.output_size[0]    # 7
            representation_size = 1024
            pose_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        representation_size = 1024
        pose_predictor = PoseRCNNPredictor(representation_size, num_classes)

        # Translation
        if translation_head is None:
            translation_head = MLPFeatureExtractor()

        if translation_predictor is None:
            translation_predictor = MLPCONCATPredictor(representation_size)

        super(PoseRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        self.roi_heads = MyRoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            pose_roi_pool=pose_roi_pool,
            pose_head=pose_head,
            pose_predictor=pose_predictor,
            translation_head=translation_head,
            translation_predictor=translation_predictor)


def posercnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=2, pretrained_backbone=True, **kwargs):

    if pretrained:
        pretrained_backbone = False

    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = PoseRCNN(backbone, num_classes, **kwargs)

    return model
