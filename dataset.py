import torch
from os.path import join
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from math import cos, sin
from transforms import get_transform
import cv2
import math
import random
import json
from car_models import car_id2name


if os.name == 'nt':
    # root_dir = '../pose_estimation'
    root_dir = 'D:/pku-autonomous-driving'
else:
    root_dir = os.path.join(os.path.expanduser("~"), 'data', 'pku-autonomous-driving')

if os.name == 'nt':
    map_root_dir = 'D:/3d-car-understanding-test/test/images'
else:
    map_root_dir = os.path.join(os.path.expanduser("~"), 'data', '3d-car-understanding-test/test/images')

if os.path.exists('a.txt'):
    os.remove('a.txt')


class PUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', debug=False):
        self.root_dir = data_dir
        self.split = split
        self.transforms = {'train': get_transform(True),
                           'val': get_transform(False),
                           'test': get_transform(False),
                           }
        self.test_file_map = dict()
        self.test_file_mirror_map = dict()

        self.data = list()
        self.indices = None
        self.load_data()
        self.w = 3384
        self.h = 2710
        # 1686.2379、1354.9849为主点坐标（相对于成像平面）
        # 摄像机分辨率 3384*2710
        self.cx = 1686.2379
        self.cy = 1354.9849
        self.fx = 2304.5479
        self.fy = 2305.8757
        self.k = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.debug = debug
        # with open('models.json', 'r') as f:
        #     self.model_size_dict = json.load(f)

    def load_data(self):
        if self.split in ['train', 'val']:
            data = pd.read_csv(join(self.root_dir, 'train.csv'))
            for (ImageId, PredictionString) in data.values:
                self.data.append({'ImageId': ImageId,
                                  'PredictionString': PredictionString})

        else:
            data = pd.read_csv(join(self.root_dir, 'sample_submission.csv'))
            for (ImageId, PredictionString) in data.values:
                self.data.append({'ImageId': ImageId,
                                  'PredictionString': PredictionString})
        sample_count = len(self.data)  # 训练集中样本的数量
        indices = np.arange(sample_count)
        np.random.seed(0)  # 固定随机种子
        np.random.shuffle(indices)

        if self.split == 'train':
            self.indices = indices[:sample_count // 40 * 39]
        elif self.split == 'val':
            self.indices = indices[-sample_count // 40:]
        else:
            self.indices = indices

        random.shuffle(self.indices)
        # 加载test map 文件
        with open('test_map.txt', 'r') as f:
            for line in f.readlines():
                pku_file, apollo_file = line.strip('\n').split()
                apollo_file = apollo_file[:-5] + '5.jpg'
                self.test_file_map[pku_file] = apollo_file

        with open('test_mirror_map.txt', 'r') as f:
            for line in f.readlines():
                pku_file, apollo_file = line.strip('\n').split()
                apollo_file = apollo_file[:-5] + '5.jpg'
                self.test_file_mirror_map[pku_file] = apollo_file

    def __getitem__(self, index):

        if self.split != 'test':
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def get_train_item(self, index):
        try:
            index_ = self.indices[index]
            sample_info = self.data[index_]
            ImageId, PredictionString = sample_info['ImageId'], sample_info['PredictionString']
            items = PredictionString.split(' ')
            model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
            rgb_path = join(self.root_dir, 'train_images', f'{ImageId}.jpg')
            # mask_path = rgb_path.replace('images', 'masks')
            image = Image.open(rgb_path)
            # flip_image = ImageOps.mirror(Image.open(rgb_path))
            image_array = np.array(image)

            # mask = Image.new('RGB', image.size)
            # try:
            #     mask = Image.open(mask_path)
            # except Exception as e:
            #     pass

            num_objs = len(model_types)
            boxes = []
            poses = []
            translations = []
            # flip_poses = []
            overlay = np.zeros_like(image_array)

            for model_type, yaw, pitch, roll, x, y, z in zip(model_types, yaws, pitches, rolls, xs, ys, zs):
                yaw, pitch, roll, xw, yw, zw = [float(x) for x in [yaw, pitch, roll, x, y, z]]
                yaw, pitch, roll = -pitch, -yaw, -roll  # 好像要变换一下

                with open(os.path.join(self.root_dir, 'car_models_json',
                                       car_id2name[int(model_type)].name + '.json')) as json_file:
                    data = json.load(json_file)
                    vertices = np.array(data['vertices'])
                    vertices[:, 1] = -vertices[:, 1]
                    triangles = np.array(data['faces']) - 1

                theta_l = yaw - math.atan2(xw, zw)
                # mirror
                theta_l_flip = -yaw - math.atan2(-xw, zw)


                Rt = np.eye(4)
                t = np.array([xw, yw, zw])
                Rt[:3, 3] = t
                Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
                Rt = Rt[:3, :]
                P = np.ones((vertices.shape[0], vertices.shape[1]+1))
                P[:, :-1] = vertices
                P[-1] = np.array([0, 0, 0, 1])
                P = P.T
                img_cor_points = np.dot(self.k, np.dot(Rt, P))
                img_cor_points = img_cor_points.T
                img_cor_points[:, 0] /= img_cor_points[:, 2]
                img_cor_points[:, 1] /= img_cor_points[:, 2]
                draw_obj(overlay, img_cor_points, triangles)

                if self.debug:
                    # print(roll)
                    print(self.image_2_world(img_cor_points[-1]),  xw, yw, zw)    # 2d转3d

                img_cor_points = img_cor_points.astype(int)

                center_x, center_y = img_cor_points[-1, :2]
                # if not check_valid(mask, center_x, center_y):
                #     if self.debug:
                #         print('跳过')
                #     continue

                xmin, ymin, xmax, ymax = self.cal_bbox(img_cor_points)
                """
                if (xmax-xmin)*(ymax-ymin) < 196:
                    if self.debug:
                        print('面积过小')
                    continue
                """
                boxes.append([xmin, ymin, xmax, ymax])
                poses.append([math.sin(theta_l), math.cos(theta_l), pitch/math.pi, roll/math.pi])  # zw is too big
                translations.append([xw, yw, zw])

                # flip_poses.append([math.sin(theta_l_flip), math.cos(theta_l_flip), pitch, -xw, yw, zw/10])

                if self.debug:
                    # image_array = draw_line(image_array, img_cor_points)
                    # image_array = self.draw_bbox(image_array, xmin, ymin, xmax, ymax)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.circle(image_array, (center_x, center_y), 5, (255, 0, 0), -1)
                    # cv2.circle(image_array, (1692, 2710), 15, (255, 0, 0), -1)
                    image_array = cv2.putText(image_array, str(round(roll, 3)), (center_x+20, center_y+20),
                                              font, 1, (255, 0, 0),
                                              thickness=2, lineType=cv2.LINE_AA)

                    # img_cor_points = self.world_2_image(model_type, -xw, yw, zw, -yaw, pitch, -roll)

                    # flip_image = draw_points(flip_image, img_cor_points)
                    # flip_image = draw_line(flip_image, img_cor_points)
                    # xmin, ymin, xmax, ymax = cal_bbox(img_cor_points)
                    # flip_image = draw_bbox(flip_image, xmin, ymin, xmax, ymax)

            if self.debug:
                alpha = 0.3
                # cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0, image_array)
                # mask = np.array(mask)
                # image_array[np.where((mask == [255, 255, 255]).all(axis=2))] = [255, 255, 255]
                image_array = cv2.resize(image_array, (1686, 1354))[:, :, ::-1]
                # if ImageId + '.jpg' in deleted_files:
                #     print(ImageId, '要被删除')
                cv2.imshow("", image_array)
                cv2.waitKey(0)
                # cv2.imwrite(f'image/{ImageId}.jpg', image_array)
                # else:
                #     cv2.imshow('', image_array)
                # print(ImageId)
                # cv2.imshow('flip', cv2.resize(flip_image, (1686, 1354))[:, :, ::-1])
                # cv2.waitKey(550)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            poses = torch.as_tensor(poses, dtype=torch.float32)
            translations = torch.as_tensor(translations, dtype=torch.float32)
            # flip_poses = torch.as_tensor(flip_poses, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor(index)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = dict()
            target["poses"] = poses
            target["translations"] = translations
            # target["flip_poses"] = flip_poses
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                image, target = self.transforms[self.split](image, target)
            return image, target

        except Exception as e:
            print(e)
            return self.get_train_item(random.randrange(0, self.__len__()))

    def get_test_item(self, index):
        # try:
        print(index)

        index = self.indices[index]
        sample_info = self.data[index]
        image_id, _ = sample_info['ImageId'], sample_info['PredictionString']
        mirror = False
        image_filename = f'{image_id}.jpg'
        rgb_path = join(self.root_dir, 'test_images', f'{image_id}.jpg')
        mask_path = rgb_path.replace('images', 'masks')
        if image_filename in self.test_file_map and os.path.exists(
                join(map_root_dir, self.test_file_map[image_filename])):
            print('redirect', image_id)
            rgb_path = join(map_root_dir, self.test_file_map[image_filename])

        elif image_filename in self.test_file_mirror_map and os.path.exists(
                join(map_root_dir, self.test_file_mirror_map[image_filename])):
            print('mirror', image_id)
            rgb_path = join(map_root_dir, self.test_file_mirror_map[image_filename])
            mirror = True
        else:
            print('不重定向', image_id)
            with open('a.txt', 'a') as f:
                print(image_id, '有问题')
                f.write(image_id + '\n')
            rgb_path = join(self.root_dir, 'test_images', f'{image_id}.jpg')

        image = Image.open(rgb_path)
        mask = Image.new('RGB', image.size)
        try:
            mask = Image.open(mask_path)
            if mirror:
                mask = ImageOps.mirror(mask)
        except Exception as e:
            pass

        if self.debug:
            cv2.imshow('', cv2.resize(np.array(image)[:, :, ::-1], (640, 480)))
            cv2.imshow('original',
                       cv2.resize(cv2.imread(join(self.root_dir, 'test_images', f'{image_id}.jpg')), (640, 480)))
            key = cv2.waitKey(0)
            if key != 32:
                with open('a.txt', 'a') as f:
                    print(image_id, '有问题')
                    f.write(image_id + '\n')
            # if mirror:
            #     cv2.destroyAllWindows()

        image_tensor, _ = self.transforms[self.split](image, _)
        return image_id, Image.open(join(self.root_dir, 'test_images', f'{image_id}.jpg')), image_tensor, mask, mirror
        # except Exception as e:
        #     print(e)

    def __len__(self):
        # if os.name == 'nt' and not self.debug:
        #     return 3
        return len(self.indices)

    def world_2_image(self, model_type, xw, yw, zw, yaw, pitch, roll):
        x_l, y_l, z_l = self.model_size_dict[model_type]
        Rt = np.eye(4)
        t = np.array([xw, yw, zw])
        Rt[:3, 3] = t
        rot_mat = euler_to_Rot(yaw, pitch, roll).T
        #
        Rt[:3, :3] = rot_mat
        Rt = Rt[:3, :]
        rotation_vec, _ = cv2.Rodrigues(Rt[:3, :3])
        # print(yaw, pitch, roll, rotation_vec, zw/10)

        P = np.array([[0, 0, 0, 1],
                      [x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],

                      [0, 0, z_l, 1],
                      [0, 0, -z_l, 1],
                      ]).T
        img_cor_points = np.dot(self.k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        return img_cor_points

    def image_2_world(self, img_cor):
        u, v, z = img_cor
        xw = (u-self.cx)/self.fx*z
        yw = (v-self.cy)/self.fy*z
        return xw, yw, z

    def cal_bbox(self, points):
        xmin, ymin, zmin = np.min(points, axis=0)
        xmax, ymax, zmax = np.max(points, axis=0)
        # xmin = np.clip(xmin, 0, self.w)
        # xmax = np.clip(xmax, 0, self.w)
        #
        # ymin = np.clip(ymin, 0, self.h)
        # ymax = np.clip(ymax, 0, self.h)
        return xmin, ymin, xmax, ymax

    def draw_bbox(self, image, xmin, ymin, xmax, ymax):
        image = np.array(image)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)

        if xmin == 0:
            cv2.line(image, (xmin, ymin), (xmin, ymax), (255, 0, 0), thickness=10)
        if xmax == self.w:
            cv2.line(image, (xmax, ymin), (xmax, ymax), (255, 0, 0), thickness=10)

        return image

def draw_obj(image, vertices, triangles):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
#         cv2.fillConvexPoly(image, coord, (0,0,255))
        cv2.polylines(image, np.int32([coord]), 1, (0,0,255))



def check_valid(mask, x, y):
    # print(mask.size, x, y)
    try:
        r, g, b = mask.getpixel((int(x), int(y)))
        if (r, g, b) == (255, 255, 255):
            return False
    except:
        pass

    return True








def draw_line(image, points):
    image = np.array(image)
    color = (255, 0, 0)
    lineTpye = cv2.LINE_4
    # cv2.line(image, tuple(points[0][:2]), tup        le(points[9][:2]), color, lineTpye)
    # cv2.line(image, tuple(points[0][:2]), tuple(points[10][:2]), color, lineTpye)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, lineTpye)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, lineTpye)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, lineTpye)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, lineTpye)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, lineTpye)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, lineTpye)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, lineTpye)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, lineTpye)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, lineTpye)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, lineTpye)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, lineTpye)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, lineTpye)
    return image


def draw_points(image, points):
    image = np.array(image)
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), 15, (255, 0, 0), -1)
    return image


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


if __name__ == '__main__':
    max_z = 0
    dataset = PUB_Dataset(root_dir, 'train', debug=True)
    for i in range(len(dataset)):
        print(i)
        dataset[i]
        # target = dataset[i][1]
        # print(target['translations'], target['poses'])

