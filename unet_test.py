from unet_parameter.model import *
from unet_parameter.data import *
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_path=('./test/HamDB­A/')
result_path=('./unet_result/')
if not os.path.exists(result_path):
    os.makedirs(result_path)
for i in os.listdir(result_path):
    path_file = os.path.join(result_path,i)  
    if os.path.isfile(path_file):
        os.remove(path_file)
model = unet('u_net_logs/000/ep009-loss0.009-acc0.996.h5')
testGene = testGenerator(test_path)
results = model.predict_generator(testGene,len(os.listdir(test_path)),verbose=1)
saveResult(test_path,result_path,results)

