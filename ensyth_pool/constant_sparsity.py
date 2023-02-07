import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
root_folder = "./baseline/"
pruning_folder = "./pruned_models/"

model_url = os.path.join(root_folder,'cifar100_ResNet20v1_model.127.h5')
loaded_model = load_model(model_url)
    # Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
num_classes = 100
    # Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Input image dimensions.
input_shape = x_train.shape[1:]

    # Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)



class Pruned_Model:
    def __init__(self,validation_split, baseline, epochs,batch_size, target_sparsity, freq,model_id,loss, optimizer):
        self.validation_split = float(validation_split)
        self.baseline = baseline
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.target_sparsity = float(target_sparsity)
        self.end_step = 0
        self.freq = int(freq)
        self.model_seq = int(model_id)
        self.loss = loss
        self.optimizer = optimizer
        self.validation_acc = 0
        self.file_name = "NA" 

    def print_attributes(self):
        print("Validation Split:{0}".format(self.validation_split))
        print("Baseline: {0}".format(self.baseline))
        print("Epochs: {0}".format(self.epochs))
        print("BatchSize: {0}".format(self.batch_size))
        print("Target Spars: {0}".format(self.target_sparsity))
        print("Frequency: {0}".format(self.freq))
        print("Loss: {0}".format(self.loss))
        print("Optimizer: {0}".format(self.optimizer))
        


    def write_acc_to_file(self, file_url):
        with open(file_url, 'a+') as file:
            file.write("{0},{1}\r\n".format(self.file_name, round(self.validation_acc,3)))
        file.close()

    def prune_model(self):
        num_images = x_train.shape[0] * (1 - self.validation_split)
        print("Num of images:",num_images)
        print("x_train.shape[0]:",x_train.shape[0])
        print("self.validation_split",self.validation_split)
        print("batch size:",self.batch_size)
        print("batch epochs:",self.epochs)

    # Define model for pruning.
        self.print_attributes()
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=self.target_sparsity,
                                                               begin_step=0,
                                                               frequency=self.freq)
       }
        model_for_pruning = prune_low_magnitude(self.baseline, **pruning_params)
        opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
      # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer=opt,
              loss=self.loss,
              metrics=['accuracy'])
       # model_for_pruning.summary()
      #fine tune
        callbacks = [ tfmot.sparsity.keras.UpdatePruningStep(),
                  ]
        model_for_pruning.fit(x_train, y_train,
                  batch_size=self.batch_size, epochs=2, validation_split=self.validation_split,
                  callbacks=callbacks, verbose=0)
        _, model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)
        self.validation_acc = model_for_pruning_accuracy
        print('Pruned test accuracy:', model_for_pruning_accuracy)
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        self.file_name = "{0}_{1}-{2}-{3}.h5".format(self.model_seq,round(self.target_sparsity,2), self.optimizer, self.loss)
        tf.keras.models.save_model(model_for_export, os.path.join(pruning_folder,self.file_name), include_optimizer=False)
        print('Saved pruned Keras model to:', self.file_name)



# Path

def main():
   
   
    from csv import reader
    hyper_parmas_path = './constant_spar_hyperparams.csv'
    if not os.path.exists(pruning_folder):
        os.makedirs(pruning_folder)

    # skip first line i.e. read header first and then iterate over each row od csv as a list
    with open(hyper_parmas_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:

                pool_model = Pruned_Model(
                    #validation_split = 
                    0.1,
                    #baseline = 
                    loaded_model,
                    #epochs
                    row[0],
                    #batch_size
                    row[1],
                    #target_sparsity = 
                    row[4],
                    #freq
                    row[5],
                    #modelid
                    row[6],
                    #loss = 
                    row[2],
                    #optimizer = 
                    row[3]
                )
                pool_model.prune_model()

                pool_model.write_acc_to_file(os.path.join(pruning_folder,'resnet_pool.csv'))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
main()