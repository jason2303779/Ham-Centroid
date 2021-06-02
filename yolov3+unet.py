import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolov3test import * 
from yolov3_parameter.yolo3.model import *
from yolov3_parameter.yolo3.utils import * 
from unet_parameter.model import *
from unet_parameter.data import *
unet_model_path='u_net_logs/000/ep020-loss0.008-acc0.997.h5'
if __name__ == '__main__':
    model = unet(unet_model_path)
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
# %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step(b):Check the resized image and save 
    for img_name in (os.listdir(test_path)):
        ori_image=cv2.imread(test_path+'/'+img_name)
        resize_image=cv2.resize(ori_image,(416,416))
        cv2.imwrite(resize_path+'/'+img_name,resize_image)
# %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step(c):Yolov3
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
    file_location=os.getcwd()
    crop_path=os.path.join(file_location,"crop")
    yolov3_cor=open(file_location+'/yolov3_result_step_2.txt','w+')
    f_ham=open(file_location+'/yolov3_step_2_size_type.txt','w+')
    fp=open(file_location+'\yolov3_step_1_blank.txt','w+')
    wrong=open(file_location+'\yolov3_step_2_wrong.txt','w+')
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    else:
        for i in os.listdir(crop_path):
            path_file = os.path.join(crop_path,i)  
            if os.path.isfile(path_file):
                os.remove(path_file)
    with open(file_location+'\yolov3_result_step_1.txt','r')as f:
        all_test_list = f.readlines()
        for k in range(len(all_test_list)):
            if all_test_list[k].count('ham')>=1:
                for j in range(all_test_list[k].count('ham')):
                    if'score' in locals().keys():
                        tmp_score=all_test_list[k].rsplit(' ')[2+j*5]
                        if float(tmp_score)>float(score):
                            big_score=j
                        else:
                            continue
                    else:
                        score=all_test_list[k].rsplit(' ')[2]
                        big_score=j
            else:
                fp.write('\n')
                fp.write(crop_path+all_test_list[k].rsplit(' ')[0])
                blank_trigger=0
# %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step(d-f):
            single_image =os.path.join(test_path,all_test_list[k].rsplit(' ')[0])
            image =cv2.imread(single_image)
            (height,width,ch)=image.shape
            a=(height/416)
            b=(width/416)
            if'blank_trigger' not in locals().keys():
                if'big_score' in locals().keys():
                    cor=all_test_list[k].rsplit('\n')[0].rsplit(';')[big_score].rsplit(' ')[-4::]
                    middle_center_y=int(int(cor[0])+((int(cor[2])-int(cor[0]))/2))
                    middle_center_x=int(int(cor[1])+((int(cor[3].replace(';',''))-int(cor[1]))/2))
                    draw_image_path=crop_path+'/'+all_test_list[k].rsplit(' ')[0]
                    c=int(20)                                                   #%%% Step(e)
                    d=int(20)                                                   #%%%
                    xmin=int((int(cor[0])-c)*b)                                 #%%%
                    xmax=int((int(cor[2])+c)*b)                                 #%%%
                    ymin=int((int(cor[1])-d)*a)                                 #%%%
                    ymax=int((int(cor[3])+d)*a)                                 #%%%
                    ham_crop=image[int(ymin):int(ymax),int(xmin):int(xmax),:]
                    if(int(ymin-c)>0)and(int(ymax+c)>0)and((int(xmin-d)>0))and((int(xmax+d)>0)):
                        if k>=1:
                            yolov3_cor.write('\n')
                        ham_crop_path= os.path.join(crop_path,all_test_list[k].rsplit(' ')[0])
                        yolov3_cor.write(ham_crop_path)
                        BndBoxLoc=(xmin,ymin,xmax,ymax)
                        yolov3_cor.write(" "+",".join([str(e) for e in BndBoxLoc]))
                        (y,x,ch_0)=ham_crop.shape
                        if k>=1:
                            f_ham.write('\n')
                        f_ham.write(ham_crop_path)
                        f_ham.write( (' x:'+str(x)+' '+'y:'+str(y)))
                        resize_image = cv2.resize(ham_crop, (224,224))
                        cv2.imwrite(draw_image_path,resize_image)
                        del big_score,middle_center_x,middle_center_y,cor,score
                    else:
                        wrong.write('\n')
                        wrong.write(all_test_list[k].rsplit(' ')[0])
            else:
                del blank_trigger
                blank_image_path=crop_path+'/'+all_test_list[k].rsplit(' ')[0]
        f_ham.close()
        yolov3_cor.close()
        wrong.close()
        fp.close()
# %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step(g-i):
        test_path_filename= 'crop'
        result_path='unetresult'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        else:
            for i in os.listdir(result_path):
                path_file = os.path.join(result_path,i)  
                if os.path.isfile(path_file):
                    os.remove(path_file)
        testGene = testGenerator(test_path_filename)
        results = model.predict_generator(testGene,len(os.listdir(test_path_filename)),verbose=1)
        saveResult(test_path_filename,result_path,results)
        pre_mask_dir=os.path.join(file_location,"unetresult")
        save_image_dir=os.path.join(file_location,"result")
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        else:
            for i in os.listdir(save_image_dir):
                path_file = os.path.join(save_image_dir,i)  
                if os.path.isfile(path_file):
                    os.remove(path_file)
        pre_mask_image=os.listdir(pre_mask_dir)
        with open(file_location+'/yolov3_step_2_size_type.txt', 'r')as f:
            all_test_list = f.readlines()
        with open(file_location+'/yolov3_result_step_2.txt', 'r')as z:
            yolo_result = z.readlines()
        for i in range(len(pre_mask_image)):
            mask_str= pre_mask_dir+'/'+ all_test_list[i].rsplit('\n')[0].rsplit(' ')[0].rsplit('\\')[-1].replace('jpg','png')
            ori_image_type=test_path+'//'+ all_test_list[i].rsplit('\n')[0].rsplit(' ')[0].rsplit('\\')[-1]
            save_path_str=save_image_dir+'//'+all_test_list[i].rsplit('\n')[0].rsplit(' ')[0].rsplit('\\')[-1]
            ori_type=cv2.imread(ori_image_type)
            cor=yolo_result[i].rsplit('\n')[0].rsplit(' ')[1].rsplit(',')
            size = ori_type.shape
            mask_zero = np.zeros(shape=(size[0], size[1], 3), dtype=np.uint8)
            mask=cv2.imread(mask_str)
            width_a=all_test_list[i].rsplit('\n')[0].rsplit(' ')[1].rsplit(':')[1]
            height_b=all_test_list[i].rsplit('\n')[0].rsplit(' ')[2].rsplit(':')[1]
            ham_ori_crop = cv2.resize(mask, (int(width_a), int(height_b)))
            mask_zero[int(cor[1]):int(cor[3]),int(cor[0]):int(cor[2])]=ham_ori_crop
            cv2.imwrite(save_path_str,mask_zero)
# %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%