#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2

from yolox.yolox_tflite import YoloxTFLite


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='04.model/yolox_nano_float16_quantize.tflite',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path = args.image

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # カメラ準備 ###############################################################
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    yolox = YoloxTFLite(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
    )

    if image_path is not None:
        image = cv2.imread(image_path)

        # 推論実施 ##############################################################
        start_time = time.time()
        bboxes, scores, class_ids = yolox.inference(image)
        elapsed_time = time.time() - start_time
        print('Elapsed time', elapsed_time)

        # 描画 ##################################################################
        image = draw_debug(
            image,
            elapsed_time,
            score_th,
            bboxes,
            scores,
            class_ids,
        )

        cv2.imshow('YOLOX ONNX Sample', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        while True:
            start_time = time.time()

            # カメラキャプチャ ################################################
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # 推論実施 ########################################################
            bboxes, scores, class_ids = yolox.inference(frame)

            elapsed_time = time.time() - start_time

            # デバッグ描画
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                score_th,
                bboxes,
                scores,
                class_ids,
            )

            # キー処理(ESC：終了) ##############################################
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映 #########################################################
            cv2.imshow('YOLOX ONNX Sample', debug_image)

        cap.release()
        cv2.destroyAllWindows()


def draw_debug(image, elapsed_time, score_th, bboxes, scores, class_ids):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # クラスID、スコア
        score = '%.2f' % score
        text = '%s:%s' % (str(int(class_id)), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
