from model_functions import build_unet
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from sklearn.model_selection import train_test_split
import scipy.io
import wandb
import tensorflow as tf


 
wandb.init(
    # set the wandb project where this run will be logged
    project="first_training",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "Simple-UNET",
    "dataset": "MATLAB",
    "epochs": 10,
    }
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)



x=[]
path="/mnt/c/users/santi/Data_simulation/x/"

def features_data(x,path):
    for elementos in range(0,1999):
        data = scipy.io.loadmat("{}X_{}.mat".format(path, elementos))
        epsil = data['epsil']
        x.append(epsil)
    return epsil

x=features_data(x,path)
y=x 
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=42)
height=x.shape[0]
width=x.shape[1]
depth=1
input_shape=(height, width, depth)

model=build_unet(input_shape)
print('construido')
model.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy", metrics=['accuracy'])
model.summary()
#batch_size=8
#steps_per_epoch=3*len(X_train)//batch_size
#history=model.fit(X_train, Y_train, batch_size = 32, epochs = 10)
#loss=history.history["loss"]
#val_loss=history.history["val_loss"]
#epochs=range(1,len(loss)+1)

#wandb.log({"loss": loss})

