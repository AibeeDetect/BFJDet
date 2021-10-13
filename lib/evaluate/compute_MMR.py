    # -*- coding:utf-8 -*-
from collections import OrderedDict
import numpy as np
import datetime
import time
from collections import defaultdict
# from . import mask as maskUtils
import copy
import matplotlib.pyplot as plt
import scipy.io as sio

import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import itertools
# from . import mask as maskUtils
import os
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

class COCO_MD:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO_MD()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['height'] = bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

class COCOMReval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  b_iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.params.imgNames = cocoGt.imgs


    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))


        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            # print(gt)
            gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1]) or \
               ( gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) or \
            ( gt['f_bbox'][3] < self.params.F_HtRng[id_setup][0] or gt['f_bbox'][3] > self.params.F_HtRng[id_setup][1]) else gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        VisRng = self.params.VisRng[id_setup]
        F_HtRng = self.params.F_HtRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, F_HtRng, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))


    def computeIoU(self, gt, dt):
        p = self.params
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g_b = [g['bbox'] for g in gt]
            g_f = [g['f_bbox'] for g in gt]
            d_b = [d['bbox'] for d in dt]
            d_f = [d['f_bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')


        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        b_ious = self.iou(d_b,g_b,iscrowd)
        f_ious = self.iou(d_f, g_f, [0 for _ in gt])

        return b_ious, f_ious

    def iou( self, dts, gts, pyiscrowd ):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2,gx2)-max(dx1,gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2,gy2)-max(dy1,gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t)/unionarea
        return ious



    def evaluateImg(self, imgId, catId, hRng, vRng, f_hRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]

        # select match pairs with face
        dt = [i for i in dt if i['f_score'] != 0.0]
        # sorted by body_score
        # dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        # sorted by (body_score + face_score)
        dtind = np.argsort([-(d['score'] + d['f_score']) for d in dt], kind='mergesort')

        dt = [dt[i] for i in dtind[0:maxDet]]
        # exclude dt out of height range

        dt = [d for d in dt if (d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter) and
              d['f_bbox'] != None and (d['f_bbox'][3] >= f_hRng[0] and d['f_bbox'][3] < f_hRng[1])]

        dtind = np.array([int(d['id'] - dt[0]['id']) for d in dt])

        # load computed ious
        if len(dtind) > 0:
            b_ious, f_ious = self.computeIoU(gt,dt)
        else:
            b_ious, f_ious = [], []

        T = len(p.b_iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        # orig method
        
        # if not len(b_ious)==0:
        #     for tind, (t, f_t) in enumerate(zip(p.b_iouThrs, p.f_iouThrs)):
        #         for dind, d in enumerate(dt):
        #             # information about best match so far (m=-1 -> unmatched)
        #             b_iou = min([t,1-1e-10])
        #             f_iou = min([f_t,1-1e-10])
        #             bstOa = (b_iou + f_iou) / 2
        #             bstg = -2
        #             bstm = -2
        #             for gind, g in enumerate(gt):
        #                 m = gtm[tind,gind]
        #                 # if this gt already matched, and not a crowd, continue
        #                 if m>0:
        #                     continue
        #                 # if dt matched to reg gt, and on ignore gt, stop
        #                 if bstm!=-2 and gtIg[gind] == 1:
        #                     break
        #                 # continue to next gt unless better match made
        #                 # if (b_ious[dind,gind] + f_ious[dind, gind]) / 2 < bstOa:
        #                 if (b_ious[dind, gind] * 0.6 + f_ious[dind, gind] * 0.4) < bstOa:
        #                     continue
        #                 # if match successful and best so far, store appropriately
        #                 # bstOa=(b_ious[dind,gind] + f_ious[dind, gind]) / 2
        #                 bstOa = b_ious[dind, gind] * 0.6 + f_ious[dind, gind] * 0.4
        #                 bstg = gind
        #                 if gtIg[gind] == 0:
        #                     bstm = 1
        #                 else:
        #                     bstm = -1

        #             # if match made store id of match for both dt and gt
        #             if bstg ==-2:
        #                 continue
        #             dtIg[tind,dind] = gtIg[bstg]
        #             dtm[tind,dind]  = gt[bstg]['id']
        #             if bstm == 1:
        #                 gtm[tind,bstg]     = d['id']
        
        if not len(b_ious)==0:
            for tind, (t, f_t) in enumerate(zip(p.b_iouThrs, p.f_iouThrs)):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    b_iou = min([t,1-1e-10])
                    f_iou = min([f_t,1-1e-10])
                    # bstOa = (b_iou + f_iou) / 2
                    bstOa_body = b_iou
                    bstOa_face = f_iou
                    bstg = -2
                    bstm = -2
                    for gind, g in enumerate(gt):
                        m = gtm[tind,gind]
                        # if this gt already matched, and not a crowd, continue
                        if m>0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if bstm!=-2 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        # if (b_ious[dind,gind] + f_ious[dind, gind]) / 2 < bstOa:
                        # if (b_ious[dind, gind] * 0.6 + f_ious[dind, gind] * 0.4) < bstOa:
                        if b_ious[dind, gind] < bstOa_body or f_ious[dind, gind] < bstOa_face:
                            continue
                        # if match successful and best so far, store appropriately
                        # bstOa=(b_ious[dind,gind] + f_ious[dind, gind]) / 2
                        # bstOa = b_ious[dind, gind] * 0.6 + f_ious[dind, gind] * 0.4
                        bstOa_body = b_ious[dind, gind]
                        bstOa_face = f_ious[dind, gind]
                        bstg = gind
                        if gtIg[gind] == 0:
                            bstm = 1
                        else:
                            bstm = -1

                    # if match made store id of match for both dt and gt
                    if bstg ==-2:
                        continue
                    dtIg[tind,dind] = gtIg[bstg]
                    dtm[tind,dind]  = gt[bstg]['id']
                    if bstm == 1:
                        gtm[tind,bstg]     = d['id']

        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'hRng':         hRng,
                'vRng':         vRng,
                'f_hRng':       f_hRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] + d['f_score'] for d in dt],
                # 'dtScores':     [d['score']for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.b_iouThrs)
        R           = len(p.fppiThrs)
        K           = len(p.catIds) if p.useCats else 1
        M           = len(p.maxDets)
        ys   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1] #_pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg==0)[1]
                tps = tps[:,inds]
                fps = fps[:,inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp)/I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]
                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except:
                        pass
                    ys[t,:,k,m] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP':   ys,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))


    def accumulate_per(self, path, id_setup, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        w = open(os.path.join(path, self.params.SetupLbl[id_setup]+'.txt'), 'w')
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.b_iouThrs)
        R           = len(p.fppiThrs)
        K           = len(p.catIds) if p.useCats else 1
        M           = len(p.maxDets)
        ys   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1] #_pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        # I0 = len(_pe.imgIds)
        I0 = 1

        # retrieve E at each category, area range, and max number of detections
        for i in i_list:
            for k, k0 in enumerate(k_list):
                Nk = k0*I0
                for m, maxDet in enumerate(m_list):
                    # E = [self.evalImgs[Nk + i] for i in i_list]
                
                    E = [self.evalImgs[Nk + i]]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue

                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.

                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    inds = np.where(dtIg==0)[1]
                    tps = tps[:,inds]
                    fps = fps[:,inds]

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fppi = np.array(fp)/I0
                        nd = len(tp)
                        recall = tp / npig
                        q = np.zeros((R,))

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        recall = recall.tolist()
                        q = q.tolist()

                        for j in range(nd - 1, 0, -1):
                            if recall[j] < recall[j - 1]:
                                recall[j - 1] = recall[j]
                        inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = recall[pi]
                        except:
                            pass
                        ys[t,:,k,m] = np.array(q)

            iouThr=0.5
            maxDets=1000
            mind = [ind for ind, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # dimension of precision: [TxRxKxAxM]
            s = ys
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.b_iouThrs)[0]
                s = s[t]
            mrs = 1-s[:,:,:,mind]

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)
            w.write(p.imgNames[i]['file_name'] + ' ' + str(round(mean_s*100, 2)) + '\n')
        w.close()

    def summarize(self,id_setup):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} | f_height={:>6s} ] = {:0.2f}%'
            titleStr = 'Miss Matching Rate'
            typeStr = '(mMR-2)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.b_iouThrs[0], p.b_iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])
            f_heightStr = '[{:0.0f}:{:0.0f}]'.format(p.F_HtRng[id_setup][0], p.F_HtRng[id_setup][1])

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.b_iouThrs)[0]
                s = s[t]
            mrs = 1-s[:,:,:,mind]

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)
            print(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, f_heightStr, mean_s*100))
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        res = _summarize(iouThr=.5,maxDets=1000)
        #_summarize(iouThr=.75,maxDets=1000)
        #_summarize(iouThr=.9,maxDets=1000)
        return res

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgNames = []
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value

        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        #self.fppiThrs= np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        #print(self.fppiThrs)
        self.fppiThrs = np.array([0.0100,    0.0178,    0.0316,    0.0562,    0.1000,    0.1778,    0.3162,    0.5623,    1.0000])
        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        # body iou阈值 和 face iou阈值设定
        self.b_iouThrs = np.array([0.5])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.f_iouThrs = np.array([0.5]) #0.3

        #self.b_iouThrs = np.array([0.75])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        #self.b_iouThrs = np.array([0.9])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        # crowdhuman
        self.HtRng = [[50, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
        self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2]]
        self.F_HtRng = [[10, 1e5 ** 2], [10, 15], [10, 1e5 ** 2], [20, 1e5 ** 2]]

        #citypersons
        # self.HtRng = [[50, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2]]
        # self.F_HtRng = [[20, 1e5 ** 2], [20, 1e5 ** 2], [20, 1e5 ** 2], [20, 1e5 ** 2]]
        # self.VisRng = [[0.65, 1e5 ** 2], [0.9, 1e5 ** 2], [0.65, 0.9], [0, 0.65]]

        #self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.002, 1e5 ** 2]]
        self.SetupLbl = ['Reasonable', 'Small','Heavy', 'All'] # crowdhuman
        # self.SetupLbl = ['Reasonable', 'Bare', 'Partial', 'Heavy'] # citypersons


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

def mmr(anno_file, json_result_file):
    mr_res_dict = OrderedDict()
    iou_type = "bbox"
    id_c = 0
    # mr_mode = "citypersons"
    mr_mode = "crowdhuman"
    for id_setup in range(3, 4):
        coco_mod_gt = COCO_MD(anno_file)
        coco_mod_dt = coco_mod_gt.loadRes(str(json_result_file))

        imgIds = sorted(coco_mod_gt.getImgIds())
        # imgIds = sorted(iduse)
        coco_mr_eval = COCOMReval(coco_mod_gt, coco_mod_dt, iou_type)
        coco_mr_eval.params.imgIds = imgIds
        coco_mr_eval.evaluate(id_setup)
        coco_mr_eval.accumulate()
        mean_s = coco_mr_eval.summarize(id_setup)
        if mr_mode == "crowdhuman":
            if id_c == 0:
                mr_res_dict["Reasonable"] = mean_s
            elif id_c == 1:
                mr_res_dict["Small"] = mean_s
            elif id_c == 2:
                mr_res_dict["Heavy"] = mean_s
            elif id_c == 3:
                mr_res_dict["All"] = mean_s
        elif mr_mode == "citypersons":
            if id_c == 0:
                mr_res_dict["Reasonable"] = mean_s
            elif id_c == 1:
                mr_res_dict["Bare"] = mean_s
            elif id_c == 2:
                mr_res_dict["Partial"] = mean_s
            elif id_c == 3:
                mr_res_dict["Heavy"] = mean_s
        id_c += 1

def compute_MMR(json_result_file, anno_file):
    mr_res_dict = OrderedDict()
    iou_type="bbox"
    id_c = 0
    mr_mode = "crowdhuman"
    for id_setup in range(3,4):
        coco_mod_gt = COCO_MD(anno_file)
        coco_mod_dt = coco_mod_gt.loadRes(str(json_result_file))

        imgIds = sorted(coco_mod_gt.getImgIds())
        # imgIds = sorted(iduse)
        coco_mr_eval = COCOMReval(coco_mod_gt, coco_mod_dt, iou_type)
        coco_mr_eval.params.imgIds = imgIds
        coco_mr_eval.evaluate(id_setup)
        coco_mr_eval.accumulate()      
        mean_s = coco_mr_eval.summarize(id_setup) 
        if mr_mode == "crowdhuman":
            if id_c == 0:
                mr_res_dict["Reasonable"] = mean_s
            elif id_c == 1:
                mr_res_dict["Small"] = mean_s
            elif id_c == 2:
                mr_res_dict["Heavy"] = mean_s
            elif id_c == 3:
                mr_res_dict["All"] = mean_s
        elif mr_mode == "citypersons":
            if id_c == 0:
                mr_res_dict["Reasonable"] = mean_s
            elif id_c == 1:
                mr_res_dict["Bare"] = mean_s
            elif id_c == 2:
                mr_res_dict["Partial"] = mean_s
            elif id_c == 3:
                mr_res_dict["Heavy"] = mean_s
        id_c += 1

