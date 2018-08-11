# coding:utf-8

"""Send JPEG image to tensorflow_model_server loaded with inception model.
python2.7
"""
#import concurrent.futures
import os
import shutil

# from __future__ import print_function

import datetime

from PIL import Image
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from diagnose_logging import Logger

# 日志
log = Logger('classify')
logger = log.getlog()


def diagnose(image_path, diagnose_name):
    try:
        host = 'localhost'
        port = '9001'
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        with open(image_path, 'rb') as im:
            image_bytes = im.read()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = diagnose_name
            request.model_spec.signature_name = 'predict_images'
            request.inputs['images'].CopyFrom(
                tf.contrib.util.make_tensor_proto(
                    image_bytes, shape=[1]))
            response = stub.Predict(request, 60.0)  # 60 secs timeout
            results = {}
            for key in response.outputs:
                tensor_proto = response.outputs[key]
                value = tf.contrib.util.make_ndarray(tensor_proto)
                results[key] = value[0][0]
            # print('classes: ' + str(results["classes"][0][0]))
            # print('scores: ' + str(results["scores"][0][0]))
            # print(results)

            classes = str(results["classes"][0][0])

            #logger.info('root_path: %s', image_path)
            logger.info('diagnose_name: %s', diagnose_name)
            logger.info('classes: %s', classes)

            return classes

    except Exception as e:
        logger.error("diagnose ERROR")
        logger.error(e)


def run_clasify(hps):
    """
    :param rootPath: 待分类原始图像文件夹名
    :param export_path_base: 输出路径根目录
    :param diagnose_name: 诊断项目名：
    """
    rootPath = hps.root
    export_path_base = hps.export
    diagnose_name = hps.diagnose

    if not os.path.exists(export_path_base):
    	os.mkdir(export_path_base)

    def run(img_path, diagnose_name):
        try:
            # 诊断结果
            img_class = diagnose(img_path, diagnose_name)
	    # 结果存储路径目录
            export_path = export_path_base + str(img_class)
            if not os.path.exists(export_path):
                os.mkdir(export_path)
            # 根据结果，分类存储
            copy_file(img_path, export_path)
        except Exception as e:
            logger.error('run ERROR')
            logger.error(e)

    for parent, dirnames, filenames in os.walk(rootPath):
        for filename in filenames:
            # 当前待处理图像完整路径
            img_path = str(parent) + '/' + str(filename)

            if is_size(img_path):
                run(img_path, diagnose_name)

                """
                # 线程池
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_list = list()
                    future_list.append(executor.submit(run, img_path, diagnose_name))
                """


def is_size(rootPath):
    """判断图像尺寸"""
    with Image.open(rootPath) as image:
        w, h = image.size
        if w > 18 and h > 18:
            return True
        else:
            return False


def copy_file(rootPath, export_path_base):
    """ 原始图像复制到类别路径下
    :param rootPath: 原图像，待分类复制
    :param export_path_base: 存储路径 + classes(类型)
    """
    # 创建类型文件夹
    try:
	"""
        if not os.path.exists(export_path_base):
            os.mkdir(export_path_base)
	"""        
	# 获得文件名
        fpath, fname = os.path.split(rootPath)
        # 复制文件
        export_path = export_path_base + '/' + str(fname)
        shutil.copyfile(rootPath, export_path)
        #shutil.move(rootPath, export_path)
        logger.info("copy %s --> %s" % (rootPath, export_path))

    except Exception as e:
        logger.error("copy_file ERROR")
        logger.error(e)


def test():
    rootPath = r'./images/example/2.jpg'
    export_path_base = './images/example/copy/'
    copy_file(rootPath, export_path_base)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="root path of data")
    parser.add_argument("--export", type=str, help="export path base")
    parser.add_argument("--diagnose", type=str, help="diagnose name")

    hps = parser.parse_args()  # So error if typo
    
    logger.info(hps.root)
    logger.info(hps.export)
    logger.info(hps.diagnose)
    
    
    print('start...')
    # 计时
    start_time = datetime.datetime.now()

    run_clasify(hps)

    end_time = datetime.datetime.now()
    time_consume_s = (end_time - start_time).seconds
   
    logger.info('start_time: %s', start_time)
    logger.info('end_time: %s', end_time)
    logger.info('time_consume: %s(s)', time_consume_s)
   
   
    
