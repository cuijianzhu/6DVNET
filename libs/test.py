from models import PoseRCNN, posercnn_resnet50_fpn, PoseRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from object_detection.dataset import PUB_Dataset, root_dir
import object_detection.utils as utils


dataset = PUB_Dataset(root_dir, 'train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn, num_workers=0)

num_classes = 2
model = posercnn_resnet50_fpn(False)    # 还没有预训练的模型, 所以先加载backbone的权重

# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# print(model)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.train()
# for images, targets in dataloader:
#     images = list(image.to(device) for image in images)
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#     loss_dict = model(images, targets)
#     print(loss_dict)

model.eval()
for images, targets in dataloader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    detections = model(images)
    print(detections)