# Find the Centroid: A Vision-based Approach for Optimal Object Grasping
Proposal of a novel pipeline that is composed of both the object detection and semantic segmentation architectures designed serially to facilitate the vision understanding process.
## Data
1) test videos -- Please download from https://drive.google.com/drive/folders/19A-p_KiBM15DNeJzNMiyfXiIixW3za7Q?usp=sharing
2) U-Net weight --Please download from https://drive.google.com/drive/folders/1wqB4iM2GovqhvDi-q9Je41Ne8q-gfjvQ
3) YOLOv3 weight --Please download from https://drive.google.com/drive/folders/1yIl55hw1UAYBHNyrQef8D9UDyx6yMBfS

The sample in the test folder is the first item above


### YOLOv3 training

* Generate your own annotation file and class name file.
* annotation file: image_file_path box1 box2 ... boxN;* 
* Box format: x_min, y_min, x_max, y_max, class_id (no spaces).
* The picture below is an example

![This is a alt text.](/show_picture/1.png "This is a sample image.")

* Change the name of the detected object in yolov3_paramete/model_data/rvoc_classes.txt
* If you want to improve the anchor , you can execute kmeans.py,then yolo_anchors.txt replace in yolov3_parameter/model_data/yolo_anchors.txt
* Change the parameters which you want to train in yolov3train.py 
```
38    batch_size = 5
39    val_split = 0.1
51    epochs=20
```
* python yolov3train.py

### U-net training

* Generate your own annotation file and class name file.
* The picture below is an example

![This is a alt text.](/show_picture/2.png "This is a sample image.")

* Change the parameters which you want to train in unet_training.py 
```
6 data_gen_args = dict(rotation_range=0.7,
7                    width_shift_range=0.5,
8                    height_shift_range=0.4,
9                   shear_range=0.4,
10                  zoom_range=0.3,
11                  horizontal_flip=True,
12                   fill_mode='nearest')
29 history =model.fit_generator(myGene,steps_per_epoch=163813,epochs=10,callbacks=[checkpoint,logging])
```
* python unet_training.py

### YOLOv3 test
* Change the parameters which you want to train in yolov3test.py 
```
15    _defaults = {
16        "model_path":'yolov3_logs/000/ep005-loss3.004-val_loss2.966.h5',
17        "anchors_path": 'yolov3_parameter/model_data/yolo_anchors.txt',
18        "classes_path": 'yolov3_parameter/model_data/voc_classes.txt',
19        "score" : 0.1,
20        "iou" : 0.5,
21        "model_image_size" : (416,416),
22        "gpu_num" : 1,
23    }
153 test_path=('./test/HamA')
```
* python yolov3test.py


### U-Net test
* Change the parameters which you want to train in yolov3test.py 
```
15    _defaults = {
16        "model_path":'yolov3_logs/000/ep005-loss3.004-val_loss2.966.h5',
17        "anchors_path": 'yolov3_parameter/model_data/yolo_anchors.txt',
18        "classes_path": 'yolov3_parameter/model_data/voc_classes.txt',
19        "score" : 0.1,
20        "iou" : 0.5,
21        "model_image_size" : (416,416),
22        "gpu_num" : 1,
23    }
153 test_path=('./test/HamA')
```

### YOLOv3+U-Net test
* Change the parameters which you want to train in yolov3+unet.py 
```
16    unet_model_path='u_net_logs/000/ep020-loss0.008-acc0.997.h5'
19    test_path=('./test/HamA')
39    resize_image=cv2.resize(ori_image,(416,416))
90    a=(height/416)
91    b=(width/416)
98    c=int(20)                                                   
99    d=int(20)
```
* python yolov3+unet



If you have suggestions or questions regarding this method, please reach out to jason2303779@yahoo.com.tw

Thank you for your interest and support.
