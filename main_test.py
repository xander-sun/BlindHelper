import sys
sys.path.append("")
import voice_recognition_server as vrs
import time

import requests
import json
import base64
import os
import logging
import speech_recognition as sr

#object detection from video
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from tensorflow.keras import backend
import zipfile
from collections import defaultdict
from io import StringIO

sys.path.append("/home/ys/train_model/models/research/object_detection")
sys.path.append("/home/ys/discovery_cup/voice_snowboy/test")
#from utils import ops as utils_ops
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

rec_words = {
            '杯子': 'cup',
            '药片': 'medicine',
            '衣服': 'clothes',
            '鞋子': 'shoes',
            '面包': 'bread',
            '牛奶': 'milk',
            '碗': 'bowl',
            '筷子': 'chopsticks',
            '充电器': 'charger',
            '化妆品': 'cosmetics',
            '椅子': 'chair',
            '门': 'door',
            '遥控器': 'remote control',
            '床': 'bed',
            '钱包': 'wallet',
            '手机': 'phone',
            '帽子': 'hat',
            '眼镜': 'glasses',
            '笔': 'pen',
            '钱': 'money',
            '卡片': 'card',
            '插座': 'socket',
            '拐棍': 'staff',
            '钥匙': 'key',
            '雨伞': 'umbrella',
            '狗': 'dog'
        
        #'香蕉':'banana',
        #     '苹果':'apple',
        #     '人':'person',
        #     '手机':'phone',
        #     '衣服':'cloth'
            }

def callback_start_wav2word():

    time.sleep(0.2)
    print("start!!!")
    key_word = wav2str()
    #key_word = '香蕉'
    print(key_word)
    key_locations = find_obj_in_camera(key_word)
    print(key_locations)
    lr, lr_angle, tb = cal_location(key_locations)
    print(lr, lr_angle, tb)
    str2voice_play(key_word, lr, lr_angle, tb)


def cal_location(key_locations):
    if key_locations is None:
        return None, None, None
    else:
        key_location = key_locations[0]
        center_x = (key_location[0] + key_location[2]) / 2.0
        center_y = (key_location[1] + key_location[3]) / 2.0
        lr = None
        lr_angle = 0
        tb = None
        if center_x <= 0.5:
            lr = '偏左'
            lr_angle = (0.5 - center_x) / 0.5 * 90
        else:
            lr = '偏右'
            lr_angle = (center_x - 0.5) / 0.5 * 90

        if center_y < 0.4:
            tb = '偏上'
        elif center_y >= 0.4 and center_y <= 0.6:
            tb = '居中'
        elif center_y > 0.6:
            tb = '偏下'

        return lr, int(lr_angle), tb

import pyttsx3

def str2voice_play(key_word, lr, lr_angle, tb):
    # 创建对象
    engine = pyttsx3.init()
    # 获取当前语音速率
    rate = engine.getProperty('rate')
    print(f'语音速率：{rate}')
    # 设置新的语音速率
    engine.setProperty('rate', 150)
    # 获取当前语音音量
    volume = engine.getProperty('volume')
    print(f'语音音量：{volume}')
    # 设置新的语音音量，音量最小为 0，最大为 1
    engine.setProperty('volume', 1.0)
    # 获取当前语音声音的详细信息
    voices = engine.getProperty('voices')
    #print(f'语音声音详细信息：{voices}')
    # 设置当前语音声音为女性，当前声音不能读中文
    engine.setProperty('voice', 'zh')
    # 设置当前语音声音为男性，当前声音可以读中文
    #engine.setProperty('voice', voices[0].id)
    # 获取当前语音声音
    voice = engine.getProperty('voice')
    print(f'语音声音：{voice}')
    # 语音文本
    #path = 'test.txt'
    #with open(path, encoding='utf-8') as f_name:
    #    words = str(f_name.readlines()).replace(r'\n', '')
    # 将语音文本说出来
    if lr is None:
        words = "没有找到您说的物品，请调整摄像头"
    else:
        words = key_word + "在设备" + lr + str(lr_angle) + "度角并且" + tb + "位置"
    engine.say(words)
    engine.runAndWait()
    engine.stop()


###########启动百度线上短语音识别服务############
token = vrs.fetch_token()

def wav2str():
    #### 录音并保存，静音时停止
    logging.basicConfig(level=logging.INFO)
    AUDIO_FILE = "./voice_rec.wav"     # 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
    while True:
        r = sr.Recognizer()
        #启用麦克风
        mic = sr.Microphone()
        logging.info('录音中...')
        print("录音中")
        with mic as source:
            #降噪
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open(AUDIO_FILE, "wb") as v:
            #将麦克风录到的声音保存到voice_rec.wav文件中
            v.write(audio.get_wav_data(convert_rate=16000))
        logging.info('录音结束，识别中...')
        result_str = vrs.pull_wav(AUDIO_FILE, token)
        result = json.loads(result_str)
        print(result['result'])

        for key_word in rec_words.keys():
            if key_word in result['result'][0]:
                print('识别到：',key_word)
                print(result_str)
                print(result['result'])
                return key_word

#########目标检测任务相关设置#######################
# What model to download.
MODEL_NAME = 'object_detection_ipython/ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph_v1.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('/home/ys/train_model/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = '/home/ys/discovery_cup/tbh_20210301_tfrecord/tbh_label_map.pbtxt'

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  #od_graph_def = tf.GraphDef()
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

import cv2
import time

def find_obj_in_camera(key_word):
    cap = cv2.VideoCapture(0)  #### choose camera id by  private computer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    time_init = time.time()
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:

            while True:
                time_start=time.time()
                ret, frame = cap.read()

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (num_detections, boxes, scores, classes) = sess.run(
                    [num_detections, boxes, scores, classes],
                    feed_dict={image_tensor: np.expand_dims(frame, 0)})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame, np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores), category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('object detection', cv2.resize(frame, (800, 600)))
                time_end=time.time()
                print('time cost',(time_end-time_start)*1000,'ms')
                if (time_end - time_init) >= 10:
                    cap.release()
                    cv2.destroyAllWindows()
                    return(None)
                
                Hit_rec = []
                for idx, score in enumerate(scores[0]):
                    if score >= 0.5:
                        cls = int(classes[0][idx])
                        print(idx, classes[0][idx])
                #    if key_word in category_index[cls]['name']:
                        if rec_words[key_word] in category_index[cls]['name']:
                            Hit_rec.append(boxes[0][cls])
                    else:
                        break
                if Hit_rec is not None:
                    print("find object")
                    print(Hit_rec)
                    cap.release()
                    cv2.destroyAllWindows()
                    return(Hit_rec)
                       
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return(None)

    cap.release()
    cv2.destroyAllWindows()

#####################
########语音关键词开机#################
import snowboydecoder
import signal

interrupted = False


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted

#if len(sys.argv) == 1:
#    print("Error: need to specify model name")
#    print("Usage: python demo.py your.model")
#    sys.exit(-1)

#model = sys.argv[1]
model = "voice_snowboy/test/hotword.pmdl"
# capture SIGINT signal, e.g., Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

detector = snowboydecoder.HotwordDetector(model, sensitivity=0.5)
print('Listening... Press Ctrl+C to exit')

# main loop
detector.start(detected_callback=callback_start_wav2word,
               interrupt_check=interrupt_callback,
               sleep_time=0.03)

detector.terminate()

