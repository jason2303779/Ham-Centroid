import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolov3train import * 
from yolov3_parameter.yolo3.model import *
from yolov3_parameter.yolo3.utils import * 
class YOLO(object):
    
    _defaults = {
        "model_path":'yolov3_logs/000/ep005-loss3.004-val_loss2.966.h5',
        "anchors_path": 'yolov3_parameter/model_data/yolo_anchors.txt',
        "classes_path": 'yolov3_parameter/model_data/voc_classes.txt',
        "score" : 0.1,
        "iou" : 0.5,
        "model_image_size" : (416,416),
        "gpu_num" : 1,
    }
    # print(_defaults)

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs) 
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  
        np.random.shuffle(self.colors)  
        np.random.seed(None) 
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,file):
        start = timer() 

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  

        out_boxes, out_scores, out_classes= self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img')) 
        font = ImageFont.truetype(font='yolov3_parameter/font/FiraMono-Medium.otf',
                    size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            file.write(predicted_class + ' ' + str(score) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + ';')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                  [left + i, top + i, right - i, bottom - i],
                  outline=self.colors[c])
            draw.rectangle(
                 [tuple(text_origin), tuple(text_origin + label_size)],
                 fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        return image

    def close_session(self):
        self.sess.close()



if __name__ == '__main__':
    test_path=('./test/HamA')
    result_path='yoloresult'
    resize_path='resizeresult'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        for i in os.listdir(result_path):
            path_file = os.path.join(result_path,i)  
            if os.path.isfile(path_file):
                os.remove(path_file)
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    else:
        for i in os.listdir(resize_path):
            path_file = os.path.join(resize_path,i)  
            if os.path.isfile(path_file):
                os.remove(path_file)
    for img_name in (os.listdir(test_path)):
        ori_image=cv2.imread(test_path+'/'+img_name)
        resize_image=cv2.resize(ori_image,(416,416))
        cv2.imwrite(resize_path+'/'+img_name,resize_image)
    txt_path ='yolov3_result_step_1.txt'
    file = open(txt_path,'w+')  
    yolo = YOLO()
    for pic_name in (os.listdir(resize_path)):        
        image_path = resize_path+'/'+pic_name   
        file.write(pic_name+' ')  
        image = Image.open(image_path)
        r_image = yolo.detect_image(image,file)
        file.write('\n')
        image_save_path = result_path+'/'+pic_name
        r_image.save(image_save_path)
    file.close() 
    yolo.close_session()