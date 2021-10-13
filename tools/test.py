import os
import sys
import math
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Queue, Process

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
# from data.CrowdHuman import CrowdHuman
from data.CrowdHuman_json import CrowdHuman
from utils import misc_utils, nms_utils
from evaluate import compute_JI, compute_APMR
from evaluate import compute_MMR
from det_oprs.bbox_opr import Pointlist_dis, matcher
from scipy.optimize import linear_sum_assignment

MAX_VAL = 8e6

def eval_all(args, config, network):
    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    evalDir = os.path.join('../model', args.model_dir, config.eval_dir)
    misc_utils.ensure_dir(evalDir)
    if 'pth' not in args.resume_weights:
        model_file = os.path.join(saveDir, 
                'dump-{}.pth'.format(args.resume_weights))
    else:
        model_file = args.resume_weights
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
    # load data
    crowdhuman = CrowdHuman(config, if_train=False)
    #crowdhuman.records = crowdhuman.records[:10]
    # multiprocessing
    num_devs = len(devices)
    len_dataset = len(crowdhuman)
    num_image = math.ceil(len_dataset / num_devs)
    result_queue = Queue(500)
    result_queue_match = Queue(500)
    procs = []
    all_results = []
    all_results_match = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, len_dataset)
        if config.network == 'pos':
            proc = Process(target=inference_pos, args=(
                    config, network, model_file, devices[i], crowdhuman, start, end, result_queue, result_queue_match))
        else:
            proc = Process(target=inference_bfj, args=(
                    config, network, model_file, devices[i], crowdhuman, start, end, result_queue, result_queue_match))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total=len_dataset, ncols=50)
    for i in range(len_dataset):
        t = result_queue.get()
        all_results.append(t)
        t_match = result_queue_match.get()
        all_results_match.extend(t_match)
        pbar.update(1)
    pbar.close()
    for p in procs:
        p.join()
    # fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    fpath = os.path.join(evalDir, 'dump-{}.json'.format(30))
    misc_utils.save_json_lines(all_results, fpath)
    fpath_match = os.path.join(evalDir, 'bf_match_bbox.json')
    misc_utils.save_json(all_results_match, fpath_match)
    # evaluation
    # res_line, JI = compute_JI.evaluation_all(fpath, 'box')
    print('processing body...')
    AP, MR = compute_APMR.compute_APMR(fpath, config.eval_source, 'box')
    line = 'BODY-->AP:{:.4f}, MR:{:.4f}.'.format(AP, MR)
    print(line)
    print('processing face...')
    AP, MR = compute_APMR.compute_APMR(fpath, config.eval_source, 'box', if_face=True)
    line = 'FACE-->AP:{:.4f}, MR:{:.4f}.'.format(AP, MR)
    print(line)
    MMR = compute_MMR.compute_MMR(fpath_match, config.eval_source)


def inference_pos(config, network, model_file, device, dataset, start, end, result_queue, result_queue_match):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # init model
    net = network()
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    # init data
    dataset.records = dataset.records[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
    # inference
    for (image, gt_boxes, im_info, ID, image_id) in data_iter:
        pred_boxes, class_num = net(image.cuda(device), im_info.cuda(device))
        scale = im_info[0, 2]
        if config.test_nms_method == 'set_nms':
            assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            top_k = pred_boxes.shape[-1] // 6
            n = pred_boxes.shape[0]
            pred_boxes = pred_boxes.reshape(-1, 6)
            idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            result = []
            for classid in range(class_num):
                keep = pred_boxes[:, 5] == (classid + 1)
                class_boxes = pred_boxes[keep]
                keep = nms_utils.set_cpu_nms(class_boxes, 0.5)
                class_boxes = class_boxes[keep]
                result.append(class_boxes)
            pred_boxes = np.vstack(result)
        elif config.test_nms_method == 'normal_nms':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            result = []
            for classid in range(class_num):
                keep = pred_boxes[:, 5] == (classid + 1)
                class_boxes = pred_boxes[keep]
                keep = nms_utils.cpu_nms(class_boxes, config.test_nms)
                class_boxes = class_boxes[keep]
                result.append(class_boxes)
            pred_boxes = np.vstack(result)
        elif config.test_nms_method == 'none':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        else:
            raise ValueError('Unknown NMS method.')
        #if pred_boxes.shape[0] > config.detection_per_image and \
        #    config.test_nms_method != 'none':
        #    order = np.argsort(-pred_boxes[:, 4])
        #    order = order[:config.detection_per_image]
        #    pred_boxes = pred_boxes[order]
        # recovery the scale
        pred_boxes[:, :4] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        gt_boxes = gt_boxes[0].numpy()
        gt_boxes[:, 2:4] -= gt_boxes[:, :2]
        match_result = match_body_face_pos(pred_boxes, image_id)
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)
        result_queue_match.put_nowait(match_result)

def inference_bfj(config, network, model_file, device, dataset, start, end, result_queue, result_queue_match):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # init model
    net = network()
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    # init data
    dataset.records = dataset.records[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
    # inference
    for (image, gt_boxes, im_info, ID, image_id) in data_iter:
        pred_boxes, pred_emb, class_num = net(image.cuda(device), im_info.cuda(device))
        scale = im_info[0, 2]
        if config.test_nms_method == 'set_nms':
            assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            top_k = pred_boxes.shape[-1] // 6
            n = pred_boxes.shape[0]
            pred_boxes = pred_boxes.reshape(-1, 6)
            idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]
        elif config.test_nms_method == 'normal_nms':
            assert pred_boxes.shape[-1] % 8 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 8)
            pred_emb = pred_emb.reshape(-1, 32)
            keep = pred_boxes[:, 6] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            pred_emb = pred_emb[keep]
            result = []
            result_emb = []
            for classid in range(class_num):
                keep = pred_boxes[:, 7] == (classid + 1)
                class_boxes = pred_boxes[keep]
                class_emb = pred_emb[keep]
                keep = nms_utils.cpu_nms(class_boxes, config.test_nms)
                class_boxes = class_boxes[keep]
                class_emb = class_emb[keep]
                result.append(class_boxes)
                result_emb.append(class_emb)
            pred_boxes = np.vstack(result)
            pred_emb = np.vstack(result_emb)
        elif config.test_nms_method == 'none':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        else:
            raise ValueError('Unknown NMS method.')
        #if pred_boxes.shape[0] > config.detection_per_image and \
        #    config.test_nms_method != 'none':
        #    order = np.argsort(-pred_boxes[:, 4])
        #    order = order[:config.detection_per_image]
        #    pred_boxes = pred_boxes[order]
        # recovery the scale
        pred_boxes[:, :6] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        gt_boxes = gt_boxes[0].numpy()
        gt_boxes[:, 2:4] -= gt_boxes[:, :2]
        match_result = match_body_face_bfj(pred_boxes, pred_emb, image_id)
        # match_result = match_body_face_bfj(pred_boxes, image_id)
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes, pred_emb), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)
        result_queue_match.put_nowait(match_result)

def match_body_face_bfj(pred_boxes, pred_emb, image_id):
    keep_body = pred_boxes[:, 7] == 1
    keep_face = pred_boxes[:, 7] == 2
    body_boxes = pred_boxes[keep_body]
    body_embs = pred_emb[keep_body]
    face_boxes = pred_boxes[keep_face]
    face_embs = pred_emb[keep_face]
    wof_flag=False

    if len(face_boxes) == 0:
        wof_flag = True
    base_body_boxes = body_boxes[:, :4]
    base_body_scores = body_boxes[:, 6]
    base_body_hooks = body_boxes[:, 4:6]

    base_face_boxes = face_boxes[:, :4]
    base_face_scores = face_boxes[:, 6]
    base_face_hooks = face_boxes[:, 4:6]

    inds_conf_base_body = (base_body_scores > 0.3).nonzero()
    if not inds_conf_base_body[0].size:
        inds_conf_base_body = np.argmax(base_body_scores)[None]
        wof_flag = True
    inds_conf_base_face = (base_face_scores > 0.3).nonzero()
    if not inds_conf_base_face[0].size and (not wof_flag):
        inds_conf_base_face = np.argmax(base_face_scores)[None]
        wof_flag = True

    base_body_boxes = base_body_boxes[inds_conf_base_body]
    base_body_hooks = base_body_hooks[inds_conf_base_body]
    base_body_scores = base_body_scores[inds_conf_base_body]
    base_body_embeddings = body_embs[inds_conf_base_body]

    if not wof_flag:
        base_face_boxes = base_face_boxes[inds_conf_base_face]
        base_face_scores = base_face_scores[inds_conf_base_face]
        base_face_hooks = base_face_hooks[inds_conf_base_face]
        base_face_embeddings = face_embs[inds_conf_base_face]

    if wof_flag:
        face_boxes = np.zeros_like(base_body_boxes)
        face_scores = np.zeros_like(base_body_scores)
    else:
        
        score_matrix = (base_face_scores[:, None] + base_body_scores) / 2

        distance_matrix = Pointlist_dis(base_face_hooks, base_body_hooks, base_body_boxes)
        embedding_matrix = np.sqrt(np.square(base_face_embeddings[:, None] - base_body_embeddings).sum(-1))
        distance_matrix_max = np.max(distance_matrix, axis=0)
        distance_matrix = distance_matrix / distance_matrix_max
        embedding_matrix_max = np.max(embedding_matrix, axis=0)
        embedding_matrix = embedding_matrix / embedding_matrix_max
        match_merge_matrix = distance_matrix * score_matrix * score_matrix + embedding_matrix * (1 - score_matrix * score_matrix)
        match_merge_matrix = np.exp(-match_merge_matrix)
        matched_vals = np.max(match_merge_matrix, axis=0)
        matched_indices = np.argmax(match_merge_matrix, axis=0)
        ignore_indices = (matched_vals < 0.98).nonzero()

        dummy_tensor = np.array([0.0, 0.0, 0.0, 0.0])

        face_boxes = base_face_boxes[matched_indices]
        face_scores = base_face_scores[matched_indices]
        if ignore_indices[0].size:

            face_boxes[ignore_indices] = dummy_tensor
            face_scores[ignore_indices] = 0
    bodylist = np.hstack((base_body_boxes, base_body_scores[:, None]))
    facelist = np.hstack((face_boxes, face_scores[:, None]))
    result = []
    for body, face in zip(bodylist, facelist):
        body = body.tolist()
        face = face.tolist()
        content = {
            'image_id': int(image_id),
            'category_id': 1,
            'bbox':[round(i, 1) for i in body[:4]],
            'score':round(float(body[4]), 5),
            'f_bbox':[round(i, 1) for i in face[:4]],
            'f_score':round(float(face[4]), 5)
        }
        result.append(content)
    return result

def match_body_face_pos(pred_boxes, image_id):
    keep_body = pred_boxes[:, 5] == 1
    keep_face = pred_boxes[:, 5] == 2
    body_boxes = pred_boxes[keep_body]
    face_boxes = pred_boxes[keep_face]
    wof_flag=False

    if len(face_boxes) == 0:
        wof_flag = True
    base_body_boxes = body_boxes[:, :4]
    base_body_scores = body_boxes[:, 4]

    base_face_boxes = face_boxes[:, :4]
    base_face_scores = face_boxes[:, 4]

    inds_conf_base_body = (base_body_scores > 0.3).nonzero()
    if not inds_conf_base_body[0].size:
        inds_conf_base_body = np.argmax(base_body_scores)[None]
        wof_flag = True
    inds_conf_base_face = (base_face_scores > 0.3).nonzero()
    if not inds_conf_base_face[0].size and (not wof_flag):
        inds_conf_base_face = np.argmax(base_face_scores)[None]
        wof_flag = True

    base_body_boxes = base_body_boxes[inds_conf_base_body]
    base_body_scores = base_body_scores[inds_conf_base_body]

    if not wof_flag:
        base_face_boxes = base_face_boxes[inds_conf_base_face]
        base_face_scores = base_face_scores[inds_conf_base_face]

    if wof_flag:
        face_boxes = np.zeros_like(base_body_boxes)
        face_scores = np.zeros_like(base_body_scores)
    else:
        body_face_distance_matrix = cal_body_face_distance_matrix(base_body_boxes, base_face_boxes)
        base_body_boxes_filter = []
        base_body_scores_filter = []
        base_face_boxes_filter = []
        base_face_scores_filter = []
        body_row_idxs, face_col_idxs = linear_sum_assignment(body_face_distance_matrix)
        for body_idx in body_row_idxs:
            f_idx = np.where(body_row_idxs == body_idx)[0][0]
            col_face_idx = face_col_idxs[f_idx]

            if body_face_distance_matrix[body_idx, col_face_idx] != MAX_VAL:
        # for body_idx in body_row_idxs:
        #     f_idx = np.where(body_row_idxs == body_idx)[0][0]
        #     col_face_idx = face_col_idxs[f_idx]
        #     if body_face_distance_matrix[body_idx, col_face_idx] != MAX_VAL:
                base_body_boxes_filter.append(base_body_boxes[body_idx])
                base_body_scores_filter.append(base_body_scores[body_idx])
                base_face_boxes_filter.append(base_face_boxes[col_face_idx])
                base_face_scores_filter.append(base_face_scores[col_face_idx])
        if base_body_boxes_filter == []:
            face_boxes = np.zeros_like(base_body_boxes)
            face_scores = np.zeros_like(base_body_scores)
            wof_flag = True
        else:
            base_body_boxes = np.vstack(base_body_boxes_filter)
            base_body_scores = np.hstack(base_body_scores_filter)
            face_boxes = np.vstack(base_face_boxes_filter)
            face_scores = np.hstack(base_face_scores_filter)

    bodylist = np.hstack((base_body_boxes, base_body_scores[:, None]))
    facelist = np.hstack((face_boxes, face_scores[:, None]))
    result = []
    for body, face in zip(bodylist, facelist):
        body = body.tolist()
        face = face.tolist()
        content = {
            'image_id': int(image_id),
            'category_id': 1,
            'bbox':[round(i, 1) for i in body[:4]],
            'score':round(float(body[4]), 5),
            'f_bbox':[round(i, 1) for i in face[:4]],
            'f_score':round(float(face[4]), 5)
        }
        result.append(content)
    return result

def cal_body_face_distance_matrix(body_boxes, face_boxes):
    body_boxes_nums = len(body_boxes)
    face_boxes_nums = len(face_boxes)
    body_face_distance_matrix = np.zeros((body_boxes_nums, face_boxes_nums))
    for body_idx in range(body_boxes_nums):
        body_box = body_boxes[body_idx]
        for face_idx in range(face_boxes_nums):
            face_box = face_boxes[face_idx]
            face_iou_in_body = one_side_iou(face_box, body_box)
            if face_iou_in_body > 0.2:
                body_face_distance_matrix[body_idx, face_idx] = 1 / face_iou_in_body
            else:
                body_face_distance_matrix[body_idx, face_idx] = MAX_VAL

    return body_face_distance_matrix

def one_side_iou(box1, box2):
    # 1. to corner box
    # box1[2:4] = box1[0:2] + box1[2:4]
    # box2[2:4] = box2[0:2] + box2[2:4]
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2] + box1[0], box2[2] + box2[0])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3] + box1[1], box2[3] + box2[1])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    # a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a1 = box1[2] * box1[3]

    iou = intersection / a1  # intersection over box 1
    return iou

def boxes_dump(boxes, embs=None):
    if boxes.shape[-1] == 8: # v2 or v3
        if embs is not None:
            result = [{'box':[round(i, 1) for i in box[:6].tolist()],
                   'score':round(float(box[6]), 5),
                   'tag':int(box[7]),
                   'emb':emb.tolist()} for box, emb in zip(boxes, embs)]
        else:
            result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                    'score':round(float(box[6]), 5),
                    'tag':int(box[7])} for box in boxes]
    elif boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]
    elif boxes.shape[-1] == 6: # v1
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]
    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--config', '-c', default=None,required=True,type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    if args.config == 'pos':
        from config_pos import config
    elif args.config == 'bfj':
        from config_bfj import config
    else:
        raise Exception('Error - only support for bfj or pos.')

    if config.network == 'pos':
        from network_pos import Network
    elif config.network == 'bfj':
        from network_bfj import Network
    else:
        raise Exception('Error - only support for bfj or pos.')
    eval_all(args, config, Network)

if __name__ == '__main__':
    run_test()

