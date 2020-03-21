#! /user/bin/env python3
# coding=utf-8
# @Time   : 2020/3/14 0014 21:45
# @Author : wangw
# @File   : run.py
# @Desc   :

from __future__ import print_function

import logging

import cv2
from PIL import Image, ImageDraw, ImageFont

from translate import Translator
translator = Translator(to_lang="chinese")

import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

config = tf.ConfigProto(allow_soft_placement=True)
# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# 开始不会给tensorflow全部gpu资源 而是按需增加

config.gpu_options.allow_growth = True

import inspect
import pickle
import random
import shutil

from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from model.agcn import Model
from feeders.feeder import Feeder

import yaml

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/kinetics/test_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, data, arg):
        self.data = data
        self.arg = arg
        self.load_model()
        self.load_data()

    def load_data(self):
        # Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(self.data, **self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        if self.arg.weights:
            # self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=[0],  # self.arg.device,
                    output_device=output_device)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def test(self, epoch, save_score=False, loader_name=['test']):

        self.model.eval()
        # self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            process = self.data_loader[ln]
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    output = self.model(data)
                    # print(output.cpu().numpy)
                    _, predict_label = torch.max(output.data, 1)
                    # print("该动作识别为为：", translator.translate(labels[predict_label.cpu().numpy()[0]]))

                    topk = torch.topk(output.data, 5).indices.cpu().numpy()[0]
                    topks = ""
                    topks_zh = ""
                    for v in topk:
                        topks = topks + ',' + labels[v]
                        topks_zh = topks_zh + ',' + labels_zh[v]

                    return topks, topks_zh

    def start(self):

        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.arg.print_log = False
        # self.print_log('Model:   {}.'.format(self.arg.model))
        # self.print_log('Weights: {}.'.format(self.arg.weights))
        action = self.test(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
        return action


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    print("components: {}".format(components))
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# def str2bool(v):
#     return v.lower() in ('yes', 'true', 't', 'y', '1')


import argparse

import time
import json


# labels = json.load(open('labels.json', 'rb'))
# print("行为中英翻译")
# labels = {v: translator.translate(k) for k, v in labels.items()}
# print(labels)
# json.dump(labels, open('lables_zh.json', 'w'))
# print("行为json文件保存")

labels_zh = json.load(open('labels-zh.json', 'rb'))
labels_zh = {v: k for k, v in labels_zh.items()}
labels = json.load(open('labels.json', 'rb'))
labels = {v: k for k, v in labels.items()}


skeleton_squence = []


class GeneratePose:

    def __init__(self):
        self.joint = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                      "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]
        self.fps_time = 0

    def gen_pose(self):
        pass

    def get_parser(self):
        parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        parser.add_argument('--camera', type=int, default=0)
        parser.add_argument('--video', type=str, default='data/openpose/video.mp4')
        parser.add_argument('--resize', type=str, default='0x0',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        parser.add_argument('--model', type=str, default='mobilenet_thin',
                            help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')

        parser.add_argument('--tensorrt', type=str, default="False",
                            help='for tensorrt process.')
        args = parser.parse_args()

        return parser

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")

        # fontStyle= ImageFont.truetype("SIMYOU.TTF", textSize, encoding="utf-8")

        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def run(self):

        # pose
        args = self.get_parser().parse_args()
        logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
        w, h = model_wh(args.resize)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), tf_config=config, trt_bool=args.tensorrt.lower() in ('yes', 'true', 't', 'y', '1'))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), tf_config=config, trt_bool=args.tensorrt.lower() in ('yes', 'true', 't', 'y', '1'))
        logger.debug('cam read+')
        # cam = cv2.VideoCapture(args.camera)
        cap = cv2.VideoCapture(args.video)

        global skeleton_squence
        frame_index = 0

        # action recognition
        parser = get_parser()

        # load arg form config file
        p = parser.parse_args()
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                    assert (k in key)
            parser.set_defaults(**default_arg)

        arg = parser.parse_args()
        init_seed(0)
        from data_gen.kinetics_gendata import gendata

        while cap.isOpened():
            # ret_val, image = cam.read()
            ret_val, image = cap.read()

            frame_index += 1

            # logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            # logger.debug("骨骼点信息 {}".format(humans))
            # print("humans {} {}".format(len(humans), humans))

            image_h, image_w = image.shape[:2]

            human_body_list = []
            for num, human in enumerate(humans):
                human_body = {}
                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                        continue
                    body_part = human.body_parts[i]
                    body_name = str(common.CocoPart(i)).split('.')[-1]
                    human_body[body_name] = {'x': body_part.x,
                                             'y': body_part.y,
                                             'score': body_part.score}

                human_body_list.append(human_body)

            pose = []
            score = []

            if human_body_list:
                for j in self.joint:
                    if j in human_body_list[0].keys():
                        pose.append(human_body_list[0][j]['x'])
                        pose.append(human_body_list[0][j]['y'])
                        score.append(human_body_list[0][j]['score'])

                    else:
                        pose.append(0)
                        pose.append(0)
                        score.append(0)
            else:
                pass

            skeleton_squence.append(
                {"frame_index": frame_index, "skeleton": [{"pose": pose, "score": score}] if pose else []})  # 单人

            if len(skeleton_squence) > 1:
                for _ in range(len(skeleton_squence) - 50):
                    skeleton_squence.pop(0)
                for i in range(len(skeleton_squence)):
                    skeleton_squence[i]['frame_index'] = i + 1
            gen_data = gendata({"data": skeleton_squence})
            processor = Processor(gen_data, arg)
            action, action_zh = processor.start()
            print("第 {} 帧 ".format(frame_index), "action:{}".format(action_zh))
            # image = self.cv2ImgAddText(image, translator.translate(action), 30, 30, (0, 255, 0), 2)
            cv2.putText(image,
                        action.split(',')[1],
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1)
            cv2.imshow('action recognition', image)
            self.fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

        # with open("open_pose_skeleton.json", 'w') as f:
        #     json.dump({"data": skeleton_squence}, f)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    GeneratePose().run()
