import os
import cv2
import torch
import numpy as np
import json

from utils import misc_utils

class COCO(object):
    def __init__(self, gt_json, dt_json = None, if_have_face = False):
        self.anno = json.load(open(gt_json, 'r'))
        self.dt_json = dt_json
        self._image_id()
        self._category_id()
        self.del_image_id()
        self._anno_process(if_have_face)
        # self._anno_process_face(if_have_face)

    def _image_id(self):
        self.image_id = {}
        self.image_wh = {}
        for _, content in enumerate(self.anno['images']):
            self.image_id[content['id']] = content['file_name']

    def del_image_id(self):
        self.image_id_new = {}
        for key, value in self.image_id.items():
            for idx, result_ann in enumerate(self.anno['annotations']):
                if result_ann['image_id'] == key:
                    if result_ann['ignore'] == 0:
                        self.image_id_new[key] = value
                        break

    def _category_id(self):
        self.category_id = {}
        for idx, category_id_name_dict in enumerate(self.anno['categories']):
            self.category_id[category_id_name_dict['id']] = category_id_name_dict['name']
        print(self.category_id)
    def _anno_process(self, if_have_face = False):
        self.annotation_id = {}
        self.annotation_face_id = {}
        for idx, result_ann in enumerate(self.anno['annotations']):
            result_attr_dict = {
                "bbox": result_ann['bbox'],
                "fbox": result_ann['f_bbox'],
                "head_center": [result_ann["h_bbox"][0]+result_ann["h_bbox"][2]/2, result_ann["h_bbox"][1]+result_ann["h_bbox"][3]/2],
                "tag": self.category_id[result_ann['category_id']],
                "extra": {'ignore': result_ann['ignore']}
            }
            if result_ann['category_id'] == 1:
                if result_ann['image_id'] not in self.annotation_id.keys():
                    self.annotation_id[result_ann['image_id']] = []
                    self.annotation_id[result_ann['image_id']].append(result_attr_dict)
                else:
                    self.annotation_id[result_ann['image_id']].append((result_attr_dict))
            else:
                if if_have_face:
                    if result_ann['image_id'] not in self.annotation_face_id.keys():
                        self.annotation_face_id[result_ann['image_id']] = []
                        self.annotation_face_id[result_ann['image_id']].append(result_attr_dict)
                    else:
                        self.annotation_face_id[result_ann['image_id']].append((result_attr_dict))
    
    def _anno_process_face(self, if_have_face = False):
        self.annotation_id = {}
        self.annotation_face_id = {}
        for idx, result_ann in enumerate(self.anno['annotations']):
            result_attr_dict = {
                "bbox": result_ann['bbox'],
                "tag": self.category_id[result_ann['category_id']],
                "extra": {'ignore': result_ann['ignore']}
            }
            if result_ann['category_id'] == 1:
                if result_ann['image_id'] not in self.annotation_id.keys():
                    self.annotation_id[result_ann['image_id']] = []
                    self.annotation_id[result_ann['image_id']].append(result_attr_dict)
                else:
                    self.annotation_id[result_ann['image_id']].append((result_attr_dict))
            else:
                if if_have_face:
                    if result_ann['image_id'] not in self.annotation_face_id.keys():
                        self.annotation_face_id[result_ann['image_id']] = []
                        self.annotation_face_id[result_ann['image_id']].append(result_attr_dict)
                    else:
                        self.annotation_face_id[result_ann['image_id']].append((result_attr_dict))

    def process_gt(self, if_have_face = False):
        result = []
        result_face = []
        for image_id, image_name in self.image_id_new.items():
            if image_id not in self.annotation_id.keys():
                continue
            image_result = {
                "ID": image_name,
                "image_id": image_id,
                "gtboxes": self.annotation_id[image_id]
            }
            result.append(image_result)
        if if_have_face:
            for image_id, image_name in self.image_id.items():
                if image_id not in self.annotation_face_id.keys():
                    gtboxes = []
                else:
                    gtboxes = self.annotation_face_id[image_id]
                image_result = {
                    "ID": image_name,
                    "image_id": image_id,
                    "gtboxes": gtboxes
                }
                result_face.append(image_result)
        return result

    def process_dt(self, if_have_face = False):
        contents = json.load(open(self.dt_json, 'r'))
        result = {}
        result_face = {}
        for content in contents:
            result.setdefault(self.image_id[content['image_id']], [])
            if not if_have_face:
                result[self.image_id[content['image_id']]].append({
                        'score': content['score'],
                        'box': content['bbox'],
                        'tag': self.category_id[content['category_id']],
                    })
            else:
                if content['category_id'] == 1:
                    result[self.image_id[content['image_id']]].append({
                        'score': content['score'],
                        'box': content['bbox'],
                        'tag': self.category_id[content['category_id']],
                    })
                else:
                    result_face.setdefault(self.image_id[content['image_id']], [])
                    result_face[self.image_id[content['image_id']]].append({
                        'score': content['score'],
                        'box': content['bbox'],
                        'tag': self.category_id[content['category_id']],
                    })
        result_process = []
        result_process_face = []
        for file_name, value in result.items():
            result_process.append(
                {
                    "ID": file_name,
                    "dtboxes": value,
                }
            )
        if if_have_face:
            for file_name, value in result_face.items():
                result_process_face.append(
                    {
                        "ID": file_name,
                        "dtboxes": value,
                    }
                )
        return result_process, result_process_face    

class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, config, if_train):
        if if_train:
            self.training = True
            source = config.train_source
            self.short_size = config.train_image_short_size
            self.max_size = config.train_image_max_size
        else:
            self.training = False
            source = config.eval_source
            self.short_size = config.eval_image_short_size
            self.max_size = config.eval_image_max_size
        # self.records = misc_utils.load_json_lines(source)
        coco = COCO(source)
        self.records = coco.process_gt()
        self.config = config

    def __getitem__(self, index):
        return self.load_record(self.records[index])

    def __len__(self):
        return len(self.records)

    def load_record(self, record):
        if self.training:
            if_flap = np.random.randint(2) == 1
        else:
            if_flap = False
        # image
        # image_path = os.path.join(self.config.image_folder, record['ID']+'.png')
        image_path = os.path.join(self.config.image_folder, record['ID'])
        image = misc_utils.load_img(image_path)
        image_h = image.shape[0]
        image_w = image.shape[1]
        if if_flap:
            image = cv2.flip(image, 1)
        if self.training:
            # ground_truth
            gtboxes = misc_utils.load_gt(record, 'gtboxes', 'bbox', self.config.class_names, self.config.network)
            keep = (gtboxes[:, 2]>=0) * (gtboxes[:, 3]>=0)
            gtboxes=gtboxes[keep, :]
            gtboxes[:, 2:4] += gtboxes[:, :2]
            if if_flap:
                gtboxes = flip_boxes(gtboxes, image_w)
            # im_info
            nr_gtboxes = gtboxes.shape[0]
            im_info = np.array([0, 0, 1, image_h, image_w, nr_gtboxes])
            return image, gtboxes, im_info
        else:
            # image
            t_height, t_width, scale = target_size(
                    image_h, image_w, self.short_size, self.max_size)
            # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
            resized_image = cv2.resize(image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
            resized_image = resized_image.transpose(2, 0, 1)
            image = torch.tensor(resized_image).float()
            gtboxes = misc_utils.load_gt(record, 'gtboxes', 'bbox', self.config.class_names, self.config.network)
            gtboxes[:, 2:4] += gtboxes[:, :2]
            gtboxes = torch.tensor(gtboxes)
            # im_info
            nr_gtboxes = gtboxes.shape[0]
            im_info = torch.tensor([t_height, t_width, scale, image_h, image_w, nr_gtboxes])
            return image, gtboxes, im_info, record['ID'], record['image_id']

    def merge_batch(self, data):
        # image
        images = [it[0] for it in data]
        gt_boxes = [it[1] for it in data]
        im_info = np.array([it[2] for it in data])
        batch_height = np.max(im_info[:, 3])
        batch_width = np.max(im_info[:, 4])
        padded_images = [pad_image(
                im, batch_height, batch_width, self.config.image_mean) for im in images]
        t_height, t_width, scale = target_size(
                batch_height, batch_width, self.short_size, self.max_size)
        # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
        resized_images = np.array([cv2.resize(
                im, (t_width, t_height), interpolation=cv2.INTER_LINEAR) for im in padded_images])
        resized_images = resized_images.transpose(0, 3, 1, 2)
        images = torch.tensor(resized_images).float()
        # ground_truth
        ground_truth = []
        for it in gt_boxes:
            gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))
            if it.shape[-1] > 5:
                it[:, 0:6] *= scale
            else:
                it[:, 0:4] *= scale
            max_box = min(self.config.max_boxes_of_image, len(it))
            gt_padded[:max_box] = it[:max_box]
            ground_truth.append(gt_padded)
        ground_truth = torch.tensor(ground_truth).float()
        # im_info
        im_info[:, 0] = t_height
        im_info[:, 1] = t_width
        im_info[:, 2] = scale
        im_info = torch.tensor(im_info)
        if max(im_info[:, -1] < 2):
            return None, None, None
        else:
            return images, ground_truth, im_info

def target_size(height, width, short_size, max_size):
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    return t_height, t_width, scale

def flip_boxes(boxes, im_w):
    flip_boxes = boxes.copy()
    for i in range(flip_boxes.shape[0]):
        flip_boxes[i, 0] = im_w - boxes[i, 2] - 1
        flip_boxes[i, 2] = im_w - boxes[i, 0] - 1
        if flip_boxes.shape[1] > 5:
            flip_boxes[i, 4] = im_w - boxes[i, 4] - 1
    return flip_boxes

def pad_image(img, height, width, mean_value):
    o_h, o_w, _ = img.shape
    margins = np.zeros(2, np.int32)
    assert o_h <= height
    margins[0] = height - o_h
    img = cv2.copyMakeBorder(
        img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    img[o_h:, :, :] = mean_value
    assert o_w <= width
    margins[1] = width - o_w
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
    img[:, o_w:, :] = mean_value
    return img
