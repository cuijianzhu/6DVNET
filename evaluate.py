import torch
from dataset import PUB_Dataset, root_dir
import utils
import math
import os
from train import get_model_detection
from eval_utils import get_acc
import numpy as np


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


@torch.no_grad()
def evaluate(model, data_loader, device, iot_threshold=0.1):

    acc_list = []

    model.eval()
    cpu_device = torch.device("cpu")

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device).numpy() for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device).numpy() for k, v in t.items()} for t in outputs]
        num_images = len(images)

        for i in range(num_images):
            pred_dict = outputs[i]
            predicted_labels = pred_dict['labels'].astype(int)
            predicted_boxes = pred_dict['boxes'].astype(int)
            predicted_scores = pred_dict['scores']
            predicted_poses = pred_dict['poses']
            for label, pred_box, score, pred_pose in zip(predicted_labels, predicted_boxes, predicted_scores,
                                                         predicted_poses):
                if score < 0.5:
                    continue
                for j, target_box in enumerate(targets[i]['boxes']):
                    iou = bb_intersection_over_union(pred_box, target_box)
                    if iou > iot_threshold:
                        pred_rot, pred_tran = get_rot_tran(pred_box, pred_pose)

                        target_pose = targets[i]['poses'][j]
                        target_rot, target_tran = get_rot_tran(target_box, target_pose)

                        # print(f'匹配成功, IoU为:{iou}', pred_box, pred_rot, pred_tran, target_rot, target_tran)
                        acc = get_acc(pred_rot+pred_tran, target_rot+target_tran)
                        # print(acc, pred_tran, target_tran)
                        acc_list.append(acc)

    print(np.array(acc_list).mean())


def get_rot_tran(box, pose):
    sin_l, cos_l, pitch, zw = pose
    u, v = (box[0] + box[2]) // 2, (box[0] + box[2])
    xw, yw, zw = dataset.image_2_world((u, v, zw*10))

    norm = sin_l ** 2 + cos_l ** 2
    sin_l = sin_l / math.sqrt(norm)
    cos_l = cos_l / math.sqrt(norm)
    theta_l = math.acos(cos_l)
    if math.asin(sin_l) < 0:
        theta_l = -theta_l
    yaw = theta_l + math.atan2(xw, zw)
    roll = math.pi
    return [yaw, pitch, roll], [xw, yw, zw]


if __name__ == '__main__':
    dataset = PUB_Dataset(root_dir=root_dir, split='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                             collate_fn=utils.collate_fn,
                                             num_workers=0 if os.name == 'nt' else 8)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_detection(num_classes=2)
    model.to(device)
    evaluate(model, dataloader, device)
