# BlindHelper

## 1.软件配置需求

1. python3及以上

2. tensorflow2.0及以上

3. 下载tensorflow下的models项目，git clone https://github.com/tensorflow/models.git, 依照https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md 说明进行目标检测object_detection项目的安装。目标检测模型的训练基于此目标检测object_detection项目。

4. 安装pyaudio

   ```
   sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
   pip install pyaudio
   ```

5. 百度AI开放平台：短语音识别-中文普通话

   https://ai.baidu.com/tech/speech

   技术文档https://ai.baidu.com/ai-doc/SPEECH/Vk38lxily

   Demo:  **https://github.com/Baidu-AIP/speech-demo**
   
6. 目前只支持Ubuntu18系统，其他系统测试中

   

## 2.使用方法

git clone git@github.com:xander-sun/BlindHelper.git

打开main_test.py, 修改其中的路径：

sys.path.append("~/models/research/object_detection")
sys.path.append("voice_snowboy/test")

目标识别模型存放路径：

MODEL_NAME = 'object_detection_ipython/ssd_mobilenet_v1_coco_2017_11_17'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('~/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

运行 main_test.py, 屏幕现实start，则程序启动成功。对话筒说启动音“苹果”（目前设置为“苹果”），则启动语音识别模式，对话筒说要寻找的物品，例如：”人在哪里？“，“香蕉在哪里？”，“找一下手机”...只要在一句话内包含可以识别的物品即可，屏幕上会显示“识别到物品”。等待1～3秒钟，程序启动目标检测任务，如果找到物品，则会在屏幕上显示“找到物品”，并通过语音提示物品方位。



## 3.模型训练

1. 程序启动音模型设置和训练

   详情查看https://github.com/xander-sun/BlindHelper/tree/main/voice_snowboy/snowboy_train

2. 语音识别模型设置和训练

   当前版本使用百度AI语音识别线上服务系统，下一版将使用自己训练的模型，并支持离线使用，当前版本不支持离线使用。

3. 目标检测模型设置和训练

   详情查看tensorflow/models下的object_detection的训练方法：https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md







