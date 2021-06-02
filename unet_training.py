from unet_parameter.model import *
from unet_parameter.data import *
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_gen_args = dict(rotation_range=0.7,
                    width_shift_range=0.5,
                    height_shift_range=0.4,
                    shear_range=0.4,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_path=('./data/')
result_path=('./unetresult/')
if not os.path.exists(result_path):
    os.makedirs(result_path)
for i in os.listdir(result_path):
    path_file = os.path.join(result_path,i)  
    if os.path.isfile(path_file):
        os.remove(path_file)
myGene = trainGenerator(5,train_path,'image','mask',data_gen_args,save_to_dir =None)
model = unet('')
log_dir = 'u_net_logs/000/'
logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, cooldown=0, min_lr=1e-10)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}.h5",
        monitor='loss',verbose=1,save_weights_only=True, save_best_only=False, period=1)
history =model.fit_generator(myGene,steps_per_epoch=163813,epochs=10,callbacks=[checkpoint,logging])
model.save_weights(log_dir + 'trained_weights.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.title("model")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["loss","acc"],loc="upper left")
plt.savefig('./u_net_training_plot.png')
plt.show()
