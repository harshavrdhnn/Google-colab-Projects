# Google-colab-Projects
Python notebooks from google colab

object_detection/
│
├── utils/
│   ├── __init__.py           # Empty file to make utils a package
│   ├── preprocessing.py      # Contains image preprocessing functions
│   └── detection.py         # Contains detection and NMS functions
│
├── models/
│   ├── __init__.py          # Empty file to make models a package
│   └── yolo.py             # YOLO model implementation
│
└── main.py                  # Main script to run the detection

Required external files (in Google Drive):
/content/drive/MyDrive/Yolo/
├── yolov3.weights           # YOLO model weights
├── yolov3.cfg              # YOLO model configuration
└── coco.names              # Class names file

/content/drive/MyDrive/Yolo/Coco/
├── custom_images/          # Directory containing your images
└── custom_annotations.json # Your COCO format annotations

