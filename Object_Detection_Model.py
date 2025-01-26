# -*- coding: utf-8 -*-

# Initialize COCO API for instance annotations
dataDir = './'
dataType = 'train2017'
annFile = './annotations/instances_' + dataType + '.json'
coco = COCO(annFile)

# To create folders with respect to several images to perform multiple/single object detection
classes = ['person', 'car', 'dog']
for cls in classes:
    os.makedirs(cls, exist_ok=True)

imgIds = coco.getImgIds()
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    img_path = os.path.join(dataDir, dataType, img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        if 'category_id' in ann:
            category = coco.loadCats(ann['category_id'])[0]['name']
            if category in classes:
                shutil.copy(img_path, os.path.join(category, img['file_name']))

from google.colab import drive
drive.mount('/content/drive')

#

# to download only required images

import os
import shutil
from pycocotools.coco import COCO
import random

# Define the number of images you want
desired_num_images = 100

# Initialize COCO API for instance annotations
dataDir = './'
dataType = 'train2017'
annFile = './annotations/instances_' + dataType + '.json'
coco = COCO(annFile)

# Create a list of image IDs containing the "person" class
catIds = coco.getCatIds(catNms=['bottle','car','person','wine glass'])
imgIds = coco.getImgIds(catIds=catIds)

# Randomly select a subset of image IDs
selected_imgIds = random.sample(imgIds, min(desired_num_images, len(imgIds)))

# To create a folder with respect to several images to perform multiple/single object detection
output_dir = 'custom_images'
os.makedirs(output_dir, exist_ok=True)

# Copy the selected images and their annotations to the output folder
for imgId in selected_imgIds:
    img = coco.loadImgs(imgId)[0]
    img_path = os.path.join(dataDir, dataType, img['file_name'])

    # Check if the image file exists before copying
    if os.path.exists(img_path):
        shutil.copy(img_path, output_dir)

        # Copy the corresponding annotation file
        ann_path = os.path.join(dataDir, 'annotations', 'instances_' + dataType + '_' + str(imgId) + '.json')
        if os.path.exists(ann_path):
            shutil.copy(ann_path, output_dir)

print("Number of images downloaded:", len(os.listdir(output_dir)))

import shutil

# Source path of the person_images folder
source_images_folder = '/content/custom_images'
# Source path of the annotations file
source_annotations_file = '/content/annotations/instances_train2017.json'

# Destination path in your Google Drive for images
destination_images_folder = '/content/drive/MyDrive/Yolo/Coco/new_custom_images'
# Destination path in your Google Drive for annotations
destination_annotations_folder = '/content/drive/MyDrive/Yolo/Coco/custom_annotations'

# Copy the person_images folder to Google Drive
shutil.copytree(source_images_folder, destination_images_folder)

# Copy the annotations file to Google Drive
# shutil.copy(source_annotations_file, destination_annotations_folder)

#Initial phase + phase 1

  import os
  import cv2
  import numpy as np
  import json
  import matplotlib.pyplot as plt
  from pycocotools.coco import COCO
  from google.colab.patches import cv2_imshow
  from sklearn.metrics import f1_score


  # Define adaptive thresholding function
  def adaptive_thresholding(image):
      # Convert image to grayscale
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Calculate mean pixel intensity
      mean_intensity = np.mean(gray_image)

      # Setting dynamic threshold based on mean intensity
      threshold = mean_intensity / 255.0  # Normalize to range [0, 1]

      return threshold

  # Define data augmentation function
  def apply_data_augmentation(image):
      # Random horizontal flip
      if random.random() > 0.5:
          image = cv2.flip(image, 1)

      # Random rotation (between -10 and 10 degrees)
      angle = random.randint(-10, 10)
      rows, cols, _ = image.shape
      M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
      image = cv2.warpAffine(image, M, (cols, rows))

      return image


      # Define bounding box refinement function
  def refine_bounding_boxes(boxes, confidences, threshold=0.5):
      refined_boxes = []
      refined_confidences = []
      if len(boxes) == 0:
          return refined_boxes, refined_confidences

      # Iterate through boxes and confidences
      for box, confidence in zip(boxes, confidences):
          if confidence > threshold:
              refined_boxes.append(box)
              refined_confidences.append(confidence)

      return refined_boxes, refined_confidences

      # Define object tracking function
  def object_tracking(frame, previous_boxes):
      # For demonstration, let's assume the tracked boxes are the same as the previous boxes
      tracked_boxes = previous_boxes
      return tracked_boxes


  # Load YOLOv3 model
  net = cv2.dnn.readNet("/content/drive/MyDrive/Yolo/yolov3.weights", "/content/drive/MyDrive/Yolo/yolov3.cfg")
  classes = []
  with open("/content/drive/MyDrive/Yolo/coco.names", "r") as f:
      classes = [line.strip() for line in f.readlines()]
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

  # COCO Parser
  def parse_coco_annotations_for_image(image_filename, annotations_data):
      images = annotations_data['images']
      annotations = annotations_data['annotations']
      categories = annotations_data['categories']

      image_id = None
      for image in images:
          if image['file_name'] == image_filename:
              image_id = image['id']
              break

      if image_id is None:
          return []  # Image not found in annotations

      category_id_to_name = {category['id']: category['name'] for category in categories}

      image_gt_labels = []
      for annotation in annotations:
          if annotation['image_id'] == image_id:
              category_id = annotation['category_id']
              category_name = category_id_to_name[category_id]
              image_gt_labels.append(category_name)

      return image_gt_labels

  # Apply non-maximum suppression (NMS)
  def apply_nms(boxes, scores, threshold=0.5):
      if len(boxes) == 0:
          return []

      # Sort boxes by their confidence scores
      indices = np.argsort(scores)[::-1]
      boxes = [boxes[i] for i in indices]

      # Initialize list to keep track of selected boxes
      selected_boxes = [boxes[0]]
      selected_indices = [indices[0]]

      # Iterate through sorted boxes
      for i in range(1, len(boxes)):
          box = boxes[i]
          box_x, box_y, box_w, box_h = box

          # Calculate intersection over union (IOU) with previously selected boxes
          iou_values = [calculate_iou(box, selected_box) for selected_box in selected_boxes]

          # Check if IOU is below threshold with all previously selected boxes
          if all(iou < threshold for iou in iou_values):
              selected_boxes.append(box)
              selected_indices.append(indices[i])

      return selected_indices

  # Calculate intersection over union (IOU) between two bounding boxes
  def calculate_iou(box1, box2):
      x1, y1, w1, h1 = box1
      x2, y2, w2, h2 = box2

      # Calculate coordinates of intersection rectangle
      x_left = max(x1, x2)
      y_top = max(y1, y2)
      x_right = min(x1 + w1, x2 + w2)
      y_bottom = min(y1 + h1, y2 + h2)

      if x_right < x_left or y_bottom < y_top:
          return 0.0

      # Calculate area of intersection rectangle
      intersection_area = (x_right - x_left) * (y_bottom - y_top)

      # Calculate area of both boxes
      box1_area = w1 * h1
      box2_area = w2 * h2

      # Calculate IOU
      iou = intersection_area / float(box1_area + box2_area - intersection_area)

      return iou

  # Load COCO annotations
  print("Loading annotations into memory...")
  try:
      coco_annotations_file = "/content/drive/MyDrive/Yolo/Coco/custom_annotations.json"
      coco = COCO(coco_annotations_file)
      print("Annotations loaded successfully!")
  except Exception as e:
      print("Error loading annotations:", e)

  # Directory containing the images
  # image_dir = "/content/drive/MyDrive/Yolo/Coco/custom_images"

  image_dir = "/content/car"
  # Limit the number of images to process
  num_images_to_process = 10

  # Counter for processed images
  num_processed_images = 0

  # Lists to store ground truth and predicted labels
  ground_truth_labels = []
  predicted_labels = []

  # Load ground truth labels from JSON file
  with open('/content/drive/MyDrive/Yolo/Coco/custom_annotations.json', 'r') as f:
      ground_truth_data = json.load(f)

  # Iterate through the images in the directory
  for filename in os.listdir(image_dir):
      # Check if the maximum number of images to process has been reached
      if num_processed_images >= num_images_to_process:
          break

      # Load image
      img_path = os.path.join(image_dir, filename)
      img = cv2.imread(img_path)

      # Check if image is loaded successfully
      if img is None:
          print(f"Error: Unable to load image {filename}")
          continue  # Skip to the next image if loading fails

      try:
          # Apply adaptive thresholding
          confidence_threshold = adaptive_thresholding(img)
          # Proceed with further processing
          height, width, channels = img.shape
          blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
          net.setInput(blob)
          outs = net.forward(output_layers)

          # Process detections
          boxes = []
          confidences = []
          class_ids = []
          for out in outs:
              for detection in out:
                  scores = detection[5:]
                  class_id = np.argmax(scores)
                  confidence = scores[class_id]
                  if confidence > 0.9:  # Confidence threshold
                      label = classes[class_id]

                      # Get coordinates of bounding box
                      center_x = int(detection[0] * width)
                      center_y = int(detection[1] * height)
                      w = int(detection[2] * width)
                      h = int(detection[3] * height)

                      # Calculate coordinates of top-left corner
                      x = int(center_x - w / 2)
                      y = int(center_y - h / 2)

                      boxes.append([x, y, w, h])
                      confidences.append(float(confidence))
                      class_ids.append(class_id)

          # Apply non-maximum suppression
          indices = apply_nms(boxes, confidences)
          detected_objects = [(classes[class_ids[i]], confidences[i]) for i in indices]

          class_labels = ["person", "bicycle","car","motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

          # Append ground truth labels
          gt_labels = parse_coco_annotations_for_image(filename, ground_truth_data)
          filtered_labels = [label for label in gt_labels if label in class_labels]
          ground_truth_labels.append(filtered_labels)

          # Append predicted labels
          predicted_labels.append([obj[0] for obj in detected_objects if obj[0] in class_labels])

          # Draw bounding boxes after applying NMS
          for i in indices:
              box = boxes[i]
              x, y, w, h = box
              label = classes[class_ids[i]]
              confidence = confidences[i]
              color = (0, 255, 0)
              cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
              cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

          # Display image with bounding boxes after applying NMS
          cv2_imshow(img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()

          # Increment the counter for processed images
          num_processed_images += 1
      except Exception as e:
          print(f"Error processing image {filename}:", e)

  filtered_ground_truth_labels = []
  for gt_labels, pred_labels in zip(ground_truth_labels, predicted_labels):
      filtered_gt_labels = sorted(list(set([label for label in gt_labels if label in pred_labels])))
      if len(filtered_gt_labels) != len(pred_labels):
          filtered_gt_labels += [''] * (len(pred_labels) - len(filtered_gt_labels))
      filtered_ground_truth_labels.append(filtered_gt_labels)

  sorted_predicted_labels = []
  for pred_labels in predicted_labels:
      sorted_pred_labels = sorted(list(set(pred_labels)))
      sorted_predicted_labels.append(sorted_pred_labels)

  # Calculate F1 score
  f1_scores = []
  for gt_labels, pred_labels in zip(filtered_ground_truth_labels, sorted_predicted_labels):
      common_labels = set(gt_labels) & set(pred_labels)
      precision = len(common_labels) / len(pred_labels) if len(pred_labels) > 0 else 0
      recall = len(common_labels) / len(gt_labels) if len(gt_labels) > 0 else 0
      if precision == 0 or recall == 0:
          f1_scores.append(0)
      else:
          f1_scores.append(2 * (precision * recall) / (precision + recall))
  avg_f1_score = sum(f1_scores) / len(f1_scores)
  print("Average F1 Score:", avg_f1_score)

  # Plot precision-recall curve
  precision_values = []
  recall_values = []
  for gt_labels, pred_labels in zip(filtered_ground_truth_labels, predicted_labels):
      true_positives = len(set(gt_labels) & set(pred_labels))
      false_positives = len(set(pred_labels) - set(gt_labels))
      false_negatives = len(set(gt_labels) - set(pred_labels))
      true_negatives = len(set(class_labels)) - (true_positives + false_positives + false_negatives)
      precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
      recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
      precision_values.append(precision)
      recall_values.append(recall)
  precision_values, recall_values = zip(*sorted(zip(precision_values, recall_values)))
  plt.plot(recall_values, precision_values, marker='o', linestyle='-')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.grid(True)
  plt.show()

#To calculate Confusion Matrix
confusion_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
for gt_labels, pred_labels in zip(ground_truth_labels, predicted_labels):
    for gt_label in gt_labels:
        if gt_label in class_labels:
            gt_index = class_labels.index(gt_label)
            for pred_label in pred_labels:
                if pred_label in class_labels:
                    pred_index = class_labels.index(pred_label)
                    confusion_matrix[gt_index][pred_index] += 1
print("Confusion Matrix:")
print(confusion_matrix)





"""08-05-Final"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from google.colab.patches import cv2_imshow
import random

# Define adaptive thresholding function
def adaptive_thresholding(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate mean pixel intensity
    mean_intensity = np.mean(gray_image)

    # Setting dynamic threshold based on mean intensity
    threshold = mean_intensity / 255.0  # Normalize to range [0, 1]

    return threshold

# Define data augmentation function
def apply_data_augmentation(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation (between -10 and 10 degrees)
    angle = random.randint(-10, 10)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    return image

# Define bounding box refinement function
def refine_bounding_boxes(boxes, confidences, threshold=0.5):
    refined_boxes = []
    refined_confidences = []
    if len(boxes) == 0:
        return refined_boxes, refined_confidences

    # Iterate through boxes and confidences
    for box, confidence in zip(boxes, confidences):
        if confidence > threshold:
            refined_boxes.append(box)
            refined_confidences.append(confidence)

    return refined_boxes, refined_confidences

# Define object tracking function
def object_tracking(frame, previous_boxes):
    # For demonstration, let's assume the tracked boxes are the same as the previous boxes
    tracked_boxes = previous_boxes
    return tracked_boxes

# Load YOLOv3 model
net = cv2.dnn.readNet("/content/drive/MyDrive/Yolo/yolov3.weights", "/content/drive/MyDrive/Yolo/yolov3.cfg")
classes = []
with open("/content/drive/MyDrive/Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO Parser
def parse_coco_annotations_for_image(image_filename, annotations_data):
    images = annotations_data['images']
    annotations = annotations_data['annotations']
    categories = annotations_data['categories']

    image_id = None
    for image in images:
        if image['file_name'] == image_filename:
            image_id = image['id']
            break

    if image_id is None:
        return []  # Image not found in annotations

    category_id_to_name = {category['id']: category['name'] for category in categories}

    image_gt_labels = []
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            category_name = category_id_to_name[category_id]
            image_gt_labels.append(category_name)

    return image_gt_labels

# Apply non-maximum suppression (NMS)
def apply_nms(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []

    # Sort boxes by their confidence scores
    indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in indices]

    # Initialize list to keep track of selected boxes
    selected_boxes = [boxes[0]]
    selected_indices = [indices[0]]

    # Iterate through sorted boxes
    for i in range(1, len(boxes)):
        box = boxes[i]
        box_x, box_y, box_w, box_h = box

        # Calculate intersection over union (IOU) with previously selected boxes
        iou_values = [calculate_iou(box, selected_box) for selected_box in selected_boxes]

        # Check if IOU is below threshold with all previously selected boxes
        if all(iou < threshold for iou in iou_values):
            selected_boxes.append(box)
            selected_indices.append(indices[i])

    return selected_indices

# Calculate intersection over union (IOU) between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# Load COCO annotations
print("Loading annotations into memory...")
try:
    coco_annotations_file = "/content/annotations/instances_train2017.json"
    coco = COCO(coco_annotations_file)
    print("Annotations loaded successfully!")
except Exception as e:
    print("Error loading annotations:", e)

# List of specific directories to process
directories_to_process = ["car", "dog"]

# Limit the number of images to process
num_images_to_process = 300

# Counter for processed images
num_processed_images = 0

# Lists to store ground truth and predicted labels
ground_truth_labels = []
predicted_labels = []

# Iterate through each directory in directories_to_process
for directory_name in directories_to_process:
    directory_path = f"/content/{directory_name}"
    for filename in os.listdir(directory_path):
        if num_processed_images >= num_images_to_process:
            break

        # Load image
        img_path = os.path.join(directory_path, filename)
        img = cv2.imread(img_path)

        # Check if image is loaded successfully
        if img is None:
            print(f"Error: Unable to load image {filename}")
            continue  # Skip to the next image if loading fails

        try:
            # Apply adaptive thresholding
            confidence_threshold = adaptive_thresholding(img)
            # Proceed with further processing
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.9:  # Confidence threshold
                        label = classes[class_id]

                        # Get coordinates of bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Calculate coordinates of top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression
            indices = apply_nms(boxes, confidences)
            detected_objects = [(classes[class_ids[i]], confidences[i]) for i in indices]

            # Append ground truth labels
            gt_labels = parse_coco_annotations_for_image(filename, ground_truth_data)
            filtered_labels = [label for label in gt_labels if label in classes]
            ground_truth_labels.append(filtered_labels)

            # Append predicted labels
            predicted_labels.append([obj[0] for obj in detected_objects if obj[0] in classes])

            # Draw bounding boxes after applying NMS
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display image with bounding boxes after applying NMS
            cv2_imshow(img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Increment the counter for processed images
            num_processed_images += 1
        except Exception as e:
            print(f"Error processing image {filename}:", e)

filtered_ground_truth_labels = []
for gt_labels, pred_labels in zip(ground_truth_labels, predicted_labels):
    filtered_gt_labels = sorted(list(set([label for label in gt_labels if label in pred_labels])))
    if len(filtered_gt_labels) != len(pred_labels):
        filtered_gt_labels += [''] * (len(pred_labels) - len(filtered_gt_labels))
    filtered_ground_truth_labels.append(filtered_gt_labels)

sorted_predicted_labels = []
for pred_labels in predicted_labels:
    sorted_pred_labels = sorted(list(set(pred_labels)))
    sorted_predicted_labels.append(sorted_pred_labels)

# Calculate F1 score
f1_scores = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, sorted_predicted_labels):
    common_labels = set(gt_labels) & set(pred_labels)
    precision = len(common_labels) / len(pred_labels) if len(pred_labels) > 0 else 0
    recall = len(common_labels) / len(gt_labels) if len(gt_labels) > 0 else 0
    if precision == 0 or recall == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * (precision * recall) / (precision + recall))
avg_f1_score = sum(f1_scores) / len(f1_scores)
print("Average F1 Score:", avg_f1_score)

# Plot precision-recall curve
precision_values = []
recall_values = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, predicted_labels):
    true_positives = len(set(gt_labels) & set(pred_labels))
    false_positives = len(set(pred_labels) - set(gt_labels))
    false_negatives = len(set(gt_labels) - set(pred_labels))
    true_negatives = len(set(classes)) - (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision_values.append(precision)
    recall_values.append(recall)
precision_values, recall_values = zip(*sorted(zip(precision_values, recall_values)))
plt.plot(recall_values, precision_values, marker='o', linestyle='-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

filtered_ground_truth_labels = []
for gt_labels, pred_labels in zip(ground_truth_labels, predicted_labels):
    filtered_gt_labels = sorted(list(set([label for label in gt_labels if label in pred_labels])))
    if len(filtered_gt_labels) != len(pred_labels):
        filtered_gt_labels += [''] * (len(pred_labels) - len(filtered_gt_labels))
    filtered_ground_truth_labels.append(filtered_gt_labels)

sorted_predicted_labels = []
for pred_labels in predicted_labels:
    sorted_pred_labels = sorted(list(set(pred_labels)))
    sorted_predicted_labels.append(sorted_pred_labels)

# Calculate F1 score
f1_scores = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, sorted_predicted_labels):
    common_labels = set(gt_labels) & set(pred_labels)
    precision = len(common_labels) / len(pred_labels) if len(pred_labels) > 0 else 0
    recall = len(common_labels) / len(gt_labels) if len(gt_labels) > 0 else 0
    if precision == 0 or recall == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * (precision * recall) / (precision + recall))
avg_f1_score = sum(f1_scores) / len(f1_scores)
print("Average F1 Score:", avg_f1_score)

# Plot precision-recall curve
precision_values = []
recall_values = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, predicted_labels):
    true_positives = len(set(gt_labels) & set(pred_labels))
    false_positives = len(set(pred_labels) - set(gt_labels))
    false_negatives = len(set(gt_labels) - set(pred_labels))
    true_negatives = len(set(classes)) - (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision_values.append(precision)
    recall_values.append(recall)
precision_values, recall_values = zip(*sorted(zip(precision_values, recall_values)))
plt.plot(recall_values, precision_values, marker='o', linestyle='-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

"""CUSTOM"""

# 03/05/24

#Initial phase + phase 1

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from google.colab.patches import cv2_imshow
from sklearn.metrics import f1_score


# Define adaptive thresholding function
def adaptive_thresholding(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate mean pixel intensity
    mean_intensity = np.mean(gray_image)

    # Setting dynamic threshold based on mean intensity
    threshold = mean_intensity / 255.0  # Normalize to range [0, 1]

    return threshold

# Define data augmentation function
def apply_data_augmentation(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation (between -10 and 10 degrees)
    angle = random.randint(-10, 10)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    return image


    # Define bounding box refinement function
def refine_bounding_boxes(boxes, confidences, threshold=0.5):
    refined_boxes = []
    refined_confidences = []
    if len(boxes) == 0:
        return refined_boxes, refined_confidences

    # Iterate through boxes and confidences
    for box, confidence in zip(boxes, confidences):
        if confidence > threshold:
            refined_boxes.append(box)
            refined_confidences.append(confidence)

    return refined_boxes, refined_confidences

    # Define object tracking function
def object_tracking(frame, previous_boxes):
    # For demonstration, let's assume the tracked boxes are the same as the previous boxes
    tracked_boxes = previous_boxes
    return tracked_boxes


# Load YOLOv3 model
net = cv2.dnn.readNet("/content/drive/MyDrive/Yolo/yolov3.weights", "/content/drive/MyDrive/Yolo/yolov3.cfg")
classes = []
with open("/content/drive/MyDrive/Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO Parser
def parse_coco_annotations_for_image(image_filename, annotations_data):
    images = annotations_data['images']
    annotations = annotations_data['annotations']
    categories = annotations_data['categories']

    image_id = None
    for image in images:
        if image['file_name'] == image_filename:
            image_id = image['id']
            break

    if image_id is None:
        return []  # Image not found in annotations

    category_id_to_name = {category['id']: category['name'] for category in categories}

    image_gt_labels = []
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            category_name = category_id_to_name[category_id]
            image_gt_labels.append(category_name)

    return image_gt_labels

# Apply non-maximum suppression (NMS)
def apply_nms(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []

    # Sort boxes by their confidence scores
    indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in indices]

    # Initialize list to keep track of selected boxes
    selected_boxes = [boxes[0]]
    selected_indices = [indices[0]]

    # Iterate through sorted boxes
    for i in range(1, len(boxes)):
        box = boxes[i]
        box_x, box_y, box_w, box_h = box

        # Calculate intersection over union (IOU) with previously selected boxes
        iou_values = [calculate_iou(box, selected_box) for selected_box in selected_boxes]

        # Check if IOU is below threshold with all previously selected boxes
        if all(iou < threshold for iou in iou_values):
            selected_boxes.append(box)
            selected_indices.append(indices[i])

    return selected_indices

# Calculate intersection over union (IOU) between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# Load COCO annotations
print("Loading annotations into memory...")
try:
    coco_annotations_file = "/content/drive/MyDrive/Yolo/Coco/custom_annotations.json"
    coco = COCO(coco_annotations_file)
    print("Annotations loaded successfully!")
except Exception as e:
    print("Error loading annotations:", e)

# Directory containing the images
image_dir = "/content/drive/MyDrive/Yolo/Coco/custom_images"

# Limit the number of images to process
num_images_to_process = 10

# Counter for processed images
num_processed_images = 0

# Lists to store ground truth and predicted labels
ground_truth_labels = []
predicted_labels = []

# Load ground truth labels from JSON file
with open('/content/drive/MyDrive/Yolo/Coco/custom_annotations.json', 'r') as f:
    ground_truth_data = json.load(f)

# Iterate through the images in the directory
for filename in os.listdir(image_dir):
    # Check if the maximum number of images to process has been reached
    if num_processed_images >= num_images_to_process:
        break

    # Load image
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)

    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image {filename}")
        continue  # Skip to the next image if loading fails

    try:
        # Apply adaptive thresholding
        confidence_threshold = adaptive_thresholding(img)
        # Proceed with further processing
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9:  # Confidence threshold
                    label = classes[class_id]

                    # Get coordinates of bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate coordinates of top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = apply_nms(boxes, confidences)
        detected_objects = [(classes[class_ids[i]], confidences[i]) for i in indices]

        class_labels = ["person", "bicycle","car","motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

        # Append ground truth labels
        gt_labels = parse_coco_annotations_for_image(filename, ground_truth_data)
        filtered_labels = [label for label in gt_labels if label in class_labels]
        ground_truth_labels.append(filtered_labels)

        # Append predicted labels
        predicted_labels.append([obj[0] for obj in detected_objects if obj[0] in class_labels])

        # Draw bounding boxes after applying NMS
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display image with bounding boxes after applying NMS
        cv2_imshow(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Increment the counter for processed images
        num_processed_images += 1
    except Exception as e:
        print(f"Error processing image {filename}:", e)

filtered_ground_truth_labels = []
for gt_labels, pred_labels in zip(ground_truth_labels, predicted_labels):
    filtered_gt_labels = sorted(list(set([label for label in gt_labels if label in pred_labels])))
    if len(filtered_gt_labels) != len(pred_labels):
        filtered_gt_labels += [''] * (len(pred_labels) - len(filtered_gt_labels))
    filtered_ground_truth_labels.append(filtered_gt_labels)

sorted_predicted_labels = []
for pred_labels in predicted_labels:
    sorted_pred_labels = sorted(list(set(pred_labels)))
    sorted_predicted_labels.append(sorted_pred_labels)

# Calculate F1 score
f1_scores = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, sorted_predicted_labels):
    common_labels = set(gt_labels) & set(pred_labels)
    precision = len(common_labels) / len(pred_labels) if len(pred_labels) > 0 else 0
    recall = len(common_labels) / len(gt_labels) if len(gt_labels) > 0 else 0
    if precision == 0 or recall == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * (precision * recall) / (precision + recall))
avg_f1_score = sum(f1_scores) / len(f1_scores)
print("Average F1 Score:", avg_f1_score)

# Plot precision-recall curve
precision_values = []
recall_values = []
for gt_labels, pred_labels in zip(filtered_ground_truth_labels, predicted_labels):
    true_positives = len(set(gt_labels) & set(pred_labels))
    false_positives = len(set(pred_labels) - set(gt_labels))
    false_negatives = len(set(gt_labels) - set(pred_labels))
    true_negatives = len(set(class_labels)) - (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision_values.append(precision)
    recall_values.append(recall)
precision_values, recall_values = zip(*sorted(zip(precision_values, recall_values)))
plt.plot(recall_values, precision_values, marker='o', linestyle='-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

