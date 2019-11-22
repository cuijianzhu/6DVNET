# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
from dataset import PUB_Dataset, root_dir
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from engine import train_one_epoch, evaluate
import utils
from libs.models import posercnn_resnet50_fpn
import sys
import argparse


parser = argparse.ArgumentParser(
    description='Faster-R-CNN Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--save_interval', type=int, default=100,
                    help='Interval to save model weights')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


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


def train():
    if os.name == 'nt':
        args.batch_size = 2
        print("running on my own xps13, so set batch_size to 1!")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_datasets = {x: PUB_Dataset(root_dir, split=x) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True,
                                                       collate_fn=utils.collate_fn,
                                                       num_workers=0 if os.name == 'nt' else 8) for x in ['train', 'val']}

    # get the model using our helper function
    model = get_model_detection(num_classes=2)

    # move model to the right device
    model.to(device)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))
    else:
        print("train from scratch")

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([{'params': params, 'initial_lr': args.lr}], lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.5, last_epoch=args.start_iter-1)

    # let's train it for 10 epochs
    num_epochs = 100
    if os.path.exists('record.txt'):
        os.remove('record.txt')

    for epoch in range(args.start_iter, args.start_iter+num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloaders_dict['train'], device, epoch, print_freq=10)
        torch.save(model.state_dict(), 'weights/epoch_%d' % epoch + '.pth')

        # # update the learning rate
        lr_scheduler.step()
        # evaluate(model,  dataloaders_dict['val'], device=device)
        #
        # sys.stdout = open("record.txt", "a")
        # print("Epoch:", epoch)
        # coco_evaluator.summarize()
        # print('-'*80)
        # sys.stdout.close()
        # sys.stdout = temp

    print("That's it!")


if __name__ == "__main__":
    train()
