# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
from dataset import PUB_Dataset, root_dir, check_valid, euler_to_Rot, draw_line
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
import utils
from libs.models import posercnn_resnet50_fpn
import sys
import cv2
import argparse
import math
import numpy as np
import json
import random
from PIL import ImageOps

parser = argparse.ArgumentParser(
    description='Faster-R-CNN Detector Testing With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=12, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str, default='weights/epoch_41.pth',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=12, type=int,
                    help='Number of workers used in dataloading')
args = parser.parse_args()


def get_model_detection(num_classes):
    model = posercnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_generator = AnchorGenerator(
        sizes=tuple([(32, 64, 128, 256, 512, 768) for _ in range(5)]),
        aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that resnet_fpn_backbone returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    return model


def test():
    if os.name == 'nt':
        args.batch_size = 1
        print("running on my own xps13, so set batch_size to 1!")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = PUB_Dataset(root_dir, 'test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              collate_fn=utils.collate_fn, num_workers=0 if os.name == 'nt' else 8)

    # get the model using our helper function
    model = get_model_detection(num_classes=2)

    # move model to the right device
    model.to(device)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=lambda storage, location: storage))
    else:
        print("Please set --resume=PATH/TO/WEIGHT")
        sys.exit(-1)

    model.eval()

    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    test_result = {}  # 保存最终的测试结果

    for i, (image_ids, images, image_tensors, masks, mirrors) in enumerate(data_loader):
        print(i, '-', len(data_loader))
        image_tensors = list(image_tensor.to(device) for image_tensor in image_tensors)
        with torch.no_grad():
            result_dict = model(image_tensors)
        num_image = len(images)
        for j in range(num_image):

            try:
                result = result_dict[j]
                image_id = image_ids[j]
                image = images[j]
                is_mirror = mirrors[j]    # 是否翻转
                mask = masks[j]
                test_result[image_id] = []
                predicted_labels = result['labels'].cpu().numpy().astype(int)
                predicted_boxes = result['boxes'].cpu().numpy().astype(int)
                predicted_scores = result['scores'].cpu().numpy()
                predicted_poses = result['poses'].cpu().numpy()
                predicted_translations = result['translations'].cpu().numpy()
                for label, box, score, pose, translation in zip(predicted_labels, predicted_boxes, predicted_scores,
                                                   predicted_poses, predicted_translations):
                    if score < 0.7:
                        continue

                    # if not check_valid(mask, center_x, center_y):
                    #     print("跳过")
                    #     continue
                    sin_l, cos_l, pitch, roll = pose
                    pitch, roll = pitch * math.pi, roll * math.pi

                    xw, yw, zw = translation
                    xw, yw, zw = xw*10, yw*10, zw*10

                    if roll > 0:
                        roll = math.pi
                    else:
                        roll = -math.pi

                    norm = sin_l ** 2 + cos_l ** 2
                    sin_l = sin_l / math.sqrt(norm)
                    cos_l = cos_l / math.sqrt(norm)
                    theta_l = math.acos(cos_l)
                    if math.asin(sin_l) < 0:
                        theta_l = -theta_l

                    # print(box, score, pose)
                    image = np.array(image)
                    # image = draw_bbox(image, xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = box
                    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2  # center_x, center_y
                    xw_, yw_, zw = dataset.image_2_world([center_x, center_y, zw])

                    if os.name == 'nt':
                        margin = 5
                        if xmin > margin and xmax < dataset.w-margin and ymin > margin and ymax < dataset.h-margin:

                            xw, yw, = xw_, yw_
                        else:
                            print('不交换', box)

                    yaw = theta_l + math.atan2(xw, zw)

                    if is_mirror:
                        print('翻转')
                        xw = -xw
                        yaw = -yaw

                        xw_ = -xw_
                        # roll = -roll #   不知道这个要不要加上去

                    print(-pitch, -yaw, -roll, xw, yw, zw, xw_, yw_, score)

                    test_result[image_id] += [str(x) for x in [-pitch, -yaw, -roll, xw, yw, zw, xw_, yw_, xmin, ymin, xmax, ymax,
                                                               score]]  # 记录一下
                    #
                    Rt = np.eye(4)
                    t = np.array([xw, yw, zw])
                    Rt[:3, 3] = t
                    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
                    Rt = Rt[:3, :]
                    P = np.array([[0, 0, 0, 1],
                                  [x_l, y_l, -z_l, 1],
                                  [x_l, y_l, z_l, 1],
                                  [-x_l, y_l, z_l, 1],
                                  [-x_l, y_l, -z_l, 1],
                                  [x_l, -y_l, -z_l, 1],
                                  [x_l, -y_l, z_l, 1],
                                  [-x_l, -y_l, z_l, 1],
                                  [-x_l, -y_l, -z_l, 1]]).T

                    img_cor_points = np.dot(dataset.k, np.dot(Rt, P)).T
                    img_cor_points[:, 0] /= img_cor_points[:, 2]
                    img_cor_points[:, 1] /= img_cor_points[:, 2]
                    img_cor_points = img_cor_points.astype(int)
                    # image = draw_points(image, img_cor_points)
                    image = draw_line(image, img_cor_points)
            except Exception as e:
                print(e)

            if os.name == 'nt':
                cv2.imshow('', cv2.resize(image, (640, 480))[:, :, ::-1])
                cv2.waitKey(0)

    with open('result.txt', 'w') as f:
        json.dump(test_result, f)


if __name__ == "__main__":
    test()
