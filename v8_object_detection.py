################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2019-2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# rm -rf doc/result/*;python3 v8_object_detection.py --input=./doc/input/NO1_highway.mp4 --output_dir=./doc/result --interval=750
#
################################################################################
import cv2
from yolov8 import YOLOv8
from imread_from_url import imread_from_url
import os, sys, math, shutil, random, datetime, signal, argparse


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def term_sig_handler(signum, frame)->None:
    sys.stdout.write('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.write('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.write('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Prepare resized images/labels dataset for LPD')
    parser.add_argument(
        "--input",
        type = str,
        required = True,
        help = "Input directory or files which you want to inference."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = True,
        help = "Ouput directory to save inference results."
    )
    parser.add_argument(
        "--model_file",
        type = str,
        required = False,
        default = './models/best.onnx',
        help = "Model file path, such as:models/best.onnx."
    )
    parser.add_argument(
        "--interval",
        type = int,
        required = False,
        default = 25,
        help = "Parse every interval frame when input is a video file."
    )
    parser.add_argument(
        "--conf_thres",
        type = float,
        required = False,
        default = 0.4,
        help = "Yolov8 conf_thres."
    )
    parser.add_argument(
        "--iou_thres",
        type = float,
        required = False,
        default = 0.3,
        help = "Yolov8 iou_thres."
    )
    parser.add_argument(
        "--target_width",
        type = int,
        required = False,
        help = "Target width for resized images/labels."
    )
    parser.add_argument(
        "--target_height",
        type = int,
        required = False,
        help = "Target height for resized images/labels."
    )
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for lop_dir1 in ("images", "labels"):
            second_dir = os.path.join(first_dir, lop_dir1)
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def inference_img(input_file, output_dir, yolov8_detector)->None:
    file_path, file_type = os.path.splitext(input_file)
    save_file = os.path.join(output_dir, file_path.split('/')[-1] + file_type)
    #print(save_file)
    img = cv2.imread(input_file, 1)
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)
    # Draw detections
    combined_img = yolov8_detector.draw_detections(img)
    cv2.imwrite(save_file, combined_img)
    return


def inference_video_file(input_file, output_dir, interval, yolov8_detector)->None:
    file_path, file_type = os.path.splitext(input_file)
    prGreen('Video file is {}'.format(input_file))
    videoCapture = cv2.VideoCapture(input_file)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    prYellow('Video info: {} {} {}'.format(fps, size, fNUMS))
    # read frames
    success = True
    #success, frame = videoCapture.read()
    loop_cnt = 0
    infer_cnt = 0
    while success:
        success, frame = videoCapture.read() #获取下一帧
        #cv2.imshow('windows', frame) #显示
        #cv2.waitKey(1000/int(fps)) #延迟
        loop_cnt += 1
        if (interval != 0) and ((loop_cnt % interval) != 0):
            continue
        # Detect Objects
        boxes, scores, class_ids = yolov8_detector(frame)
        # Draw detections
        combined_img = yolov8_detector.draw_detections(frame)
        save_file = os.path.join(output_dir, file_path.split('/')[-1] + '_' + str(infer_cnt).zfill(20) + '.jpg')
        infer_cnt += 1
        cv2.imwrite(save_file, combined_img)
        prGreen('Save inference result: {}'.format(save_file))
        #print(loop_cnt)
    prGreen('Read images count: {}, inference images count:{}'.format(loop_cnt, infer_cnt))
    videoCapture.release()
    return


def deal_input_file(input_file, output_dir, interval, yolov8_detector)->None:
    file_path, file_type = os.path.splitext(input_file)
    if file_path.split('/')[-1] == '.gitignore':
        prGreen('It is a \'.gitignore\' file, return')
        return
    if file_type in ('.jpg', '.png'):
        inference_img(input_file, output_dir, yolov8_detector)
    elif file_type in ('.mp4'):
        inference_video_file(input_file, output_dir, interval, yolov8_detector)
    else:
        prYellow('file_type({}) not support, return'.format(file_type))
        return
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    args.input = os.path.abspath(args.input)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('input: {}, output_dir: {}, model_file: {}'.format(args.input, args.output_dir, args.model_file))
    # Initialize yolov8 object detector
    yolov8_detector = YOLOv8(args.model_file, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
    #------
    if os.path.isdir(args.input):
        #print("it's a directory")
        for root, dirs, files in os.walk(args.input):
            for lop_file in files:
                deal_file = os.path.join(root, lop_file)
                #print(deal_file)
                deal_input_file(deal_file, args.output_dir, args.interval, yolov8_detector)
    elif os.path.isfile(args.input):
        #print("it's a normal file")
        deal_input_file(args.input, args.output_dir, args.interval, yolov8_detector)
    else:
        prRed(skk)("it's a special file(socket,FIFO,device file)")
        return
    return


if __name__ == "__main__":
    main_func()