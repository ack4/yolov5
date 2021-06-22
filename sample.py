import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def main():
    with torch.no_grad():
        pass
    # with torch.no_grad():
    #     hub()
    # exit(0)

    with torch.no_grad():
        src = "0"
        weights = "yolov5x6.pt"
        # weights = "yolov5s.pt"
        detect(weights, src)
        # strip_optimizer("yolov5x.pt")


def hub():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    cudnn.benchmark = True  # set True to speed up constant image size inference

    names = model.module.names if hasattr(model, 'module') else model.names
    np.random.seed(2)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Inference
    cap = cv2.VideoCapture(0)
    while True:

        _, img = cap.read()
        img = img[60:-60, 140:-140, :]
        cv2.resize(img, (640, 640))
        print(img.shape)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        continue

        img_infer = np.asarray([cv2.resize(img, (640, 640))]).astype(np.float32)
        img_infer /= 255.0
        img_infer = img_infer.transpose((0, 3, 1, 2))

        tensor = torch.from_numpy(img_infer)
        pred = model(tensor)[0]
        pred = non_max_suppression(prediction=pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

        # 1batch
        det = pred[0]

        det[:, :4] = scale_coords(img_infer.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=2)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


def detect(weights: str, src: str):
    device = select_device('')
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    infer_size = check_img_size(640, s=stride)
    if half:
        model.half()

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(src, img_size=infer_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    np.random.seed(2)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, infer_size, infer_size).to(device).type_as(next(model.parameters())))

    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        # pred, features = model(img)
        # print(pred.shape)
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)

        pred = non_max_suppression(prediction=pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

        # 1batch
        im0 = im0s[0].copy()
        det = pred[0]

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

        cv2.imshow("img", im0)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
