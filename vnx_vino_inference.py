import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
import openvino.inference_engine as ie
import cv2

import vnxpy

# import torch
# import torchvision

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

"""
    Non-max Suppression Algorithm
    @param list  Object candidate bounding boxes
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return Rest boxes after nms operation
"""
def nms(bounding_boxes, confidence_score, classes, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_class = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_class.append(classes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, np.asarray(picked_score), np.asarray(picked_class)

        
class ObjectDetector(vnxpy.Analytics1):

    def __init__(self, vv: vnxpy.Vnxvideo = None):
        super().__init__(vv)
        self.set_params()
        self.init_ie()

    def set_params(self):
        if 'skip' in self.config:
            self.skip = self.config['skip']
        else:
            self.skip = 0
        self.nframe = 0

        if 'network' in self.config:
            self.network_config = self.config['network']
        else:
            self.network_config = 'best_openvino_model_2\\best.xml'
        if 'classes' in self.config:
            self.classes = self.config['classes']
        else:
            self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']

        if 'device' in self.config:
            self.device_name = self.config['device']
        else:
            self.device_name = 'CPU'

        if 'confidence' in self.config:
            self.confidence = self.config['confidence']
        else:
            self.confidence = 0.5

        if 'iou_threshold' in self.config:
            self.iou_threshold = self.config['iou_threshold']
        else:
            self.iou_threshold = 0.45

        if 'send_raw_detections' in self.config:
            self.send_raw_detections = self.config['send_raw_detections']
        else:
            self.send_raw_detections = False

    def init_ie(self):
        self.core = ie.IECore()
        self.network = self.core.read_network(model=self.network_config,
                                              weights=Path(self.network_config).with_suffix('.bin'))
        
        self.network.input_info['images'].precision = 'U8'
        self.network.input_info['images'].preprocess_info.color_format = ie.ColorFormat.RGB
        self.network.input_info['images'].layout = 'NHWC'
        self.executable_network = self.core.load_network(self.network, device_name=self.device_name, num_requests=1)

    def onsample(self, sample : vnxpy.RawSample, timestamp):
        self.nframe = self.nframe+1
        
        if self.skip > 0 and self.nframe % self.skip != 0:
            return

        imorig = sample.rgb24()
        imorig = imorig * 255.0
        imorig = imorig.astype(np.ubyte)

        ow, oh, _ = imorig.shape
        
        im, ratio, (dw, dh) = letterbox(imorig, (640, 640), auto=False)

        im = im[np.newaxis, ...] #to batch, channel, height, width

        b, h, w, ch = im.shape  # batch, channel, height, width

        desc = ie.TensorDesc(precision='U8', dims=(b, ch, h, w), layout='NHWC')  # Tensor Description
        request = self.executable_network.requests[0]  # inference request
        request.set_blob(blob_name='images', blob=ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
        request.infer()
        y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs)) #batch, num_detects,  xywh + obj + num_classes

        #assume batch=1
        y = y[0,...]
        y[:, 5:] *= y[:, 4:5]  # conf = obj_conf * cls_conf

        #filter by confidence
        conf = y[:, 5:].max(axis=1, keepdims=True)
        j = y[:, 5:].argmax(axis=1)
        y = np.concatenate((y[:, 0:4], conf, j.astype(np.float)[..., np.newaxis]), 1)[conf[:,0] > self.confidence] # xywh, conf, class

        boxes = xywh2xyxy(y[:, :4])
        conf = y[:, 4]
        classes = y[:, 5]

        #hand_written_nms
        boxes, conf, classes = nms(boxes, conf, classes, self.iou_threshold)
        if len(boxes) == 0:
            detects = np.zeros((0,6))
        else:
            detects = np.concatenate((boxes, conf[...,np.newaxis], classes[...,np.newaxis]),1) #xyxy conf class

        #rescale back to original
        detects[:, 0] = (detects[:, 0] - dw) / ratio[0]
        detects[:, 1] = (detects[:, 1] - dh) / ratio[1]
        detects[:, 2] = (detects[:, 2] - dw) / ratio[0]
        detects[:, 3] = (detects[:, 3] - dh) / ratio[1]
        
        for i, detect in enumerate(detects):
            if self.send_raw_detections:
                self.event_from_detection(ow, oh, timestamp, detect)
            imorig = cv2.rectangle(imorig, (int(detect[0]), int(detect[1])), (int(detect[2]), int(detect[3])), (255, 0, 0), 2)
        cv2.imshow("Letter", imorig)
        cv2.waitKey(10) 
        
        print(self.nframe)

    def event_from_detection(self, w, h, timestamp, detect):
        class_name = (self.classes[int(detect[5])]) if detect[5] < len(self.classes) else detect[5]
        self.event('ObjectDetected',
                   {'left': detect[0] / w,
                    'top': detect[1] / h,
                    'right': detect[2] / w,
                    'bottom': detect[3] / h,
                    'confidence': detect[4],
                    'class': class_name,
                    'timestamp': timestamp})

ObjectDetector().run()
