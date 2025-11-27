import os
import cv2
import torch
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# Paths
weights = 'C:/Users/hp/Desktop/intern/yolov5/runs/train/exp_augmented_120/weights/best.pt'
source = 'C:/Users/hp/Desktop/intern/videos/input.mp4'
output_path = 'C:/Users/hp/Desktop/intern/videos/output.mp4'
imgsz = 832
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('predicted_labels', exist_ok=True)

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, imgsz, imgsz))

# Load video
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
vid_writer = None

# Tracker
tracker = DeepSort(max_age=30)

# Pole counts
left_pole_ids = set()
right_pole_ids = set()

frame_idx = 0  # For naming prediction files uniquely

for path, img, im0s, vid_cap, _ in dataset:
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    im0 = im0s.copy()
    if len(pred):
        det = pred[0]
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

        # Save YOLO-format predictions
        label_output_path = f"predicted_labels/frame_{frame_idx:05d}.txt"
        with open(label_output_path, "w") as f:
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                x_center = ((x1 + x2) / 2) / im0.shape[1]
                y_center = ((y1 + y2) / 2) / im0.shape[0]
                width = (x2 - x1) / im0.shape[1]
                height = (y2 - y1) / im0.shape[0]
                f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        detections = []
        for *xyxy, conf, cls in det:
            label = names[int(cls)]
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1])]
            detections.append((bbox, conf.item(), label))

        tracks = tracker.update_tracks(detections, frame=im0)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            class_name = track.get_det_class()

            if class_name == 'left_pole':
                left_pole_ids.add(track_id)
                color = (255, 0, 0)
            elif class_name == 'right_pole':
                right_pole_ids.add(track_id)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(im0, (int(l), int(t)), (int(l + w), int(t + h)), color, 2)
            cv2.putText(im0, f"{class_name} #{track_id}", (int(l), int(t) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show count
    total = len(left_pole_ids) + len(right_pole_ids)
    cv2.putText(im0, f"Left Poles: {len(left_pole_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(im0, f"Right Poles: {len(right_pole_ids)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(im0, f"Total Poles: {total}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Write output
    if vid_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    vid_writer.write(im0)

    cv2.imshow('Tracking', im0)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_idx += 1  # Move to next frame

cv2.destroyAllWindows()
if vid_writer:
    vid_writer.release()

print(f"\n Tracking video saved to: {output_path}")
print(" YOLO-format predictions saved to: predicted_labels/")
