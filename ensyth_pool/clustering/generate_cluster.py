import tensorflow as tf
import glob
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


#pruning_folder = "./test_cluster/"
pruning_folder = "./all_pruned_models/"

def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)


def get_meta_data(file_name):
    first = find_nth(file_name,'-',1)
    second = find_nth(file_name,'-',2)
    optimizer = file_name[first+1: second]
    loss = file_name[second+1:len(file_name)-3]
    return optimizer,loss


def file_browser(directory_name):
  return [model for model in glob.glob(directory_name + "*.h5")]

def intialize_train_set():
  subtract_pixel_mean = True
  num_classes = 100
  # Load the CIFAR100 data.
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
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
 
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  return x_train,y_train

def model_eval(loaded_model, x_test):
    loaded_model = tf.keras.models.load_model(loaded_model)
    model_prediction = loaded_model.predict(x_test)
    return model_prediction

def compute_acc2(x_test, y_test,keras_model):
    loaded_model = tf.keras.models.load_model(keras_model)
    opt,loss = get_meta_data(keras_model)
    loaded_model.compile(optimizer=opt,loss = loss, metrics=['accuracy'])
    scores = loaded_model.evaluate(x_test, y_test, verbose=1)
    print('test acc:',scores[1])
    return scores[1]
def affinity_ropagation_clustering(x_input):
    return AffinityPropagation(random_state=1234).fit(x_input)

def kmeans_clustering(x_input):
    return KMeans(n_clusters=20).fit(x_input)


def kmeans_elbow_clustering(x_input):
    num_clusters = range (1,1120)
    distorsions = []
    for k in num_clusters:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(x_input)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15,5))
    plt.plot(num_clusters, distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.savefig("elbow.png")



def main():
    x_test,y_test = intialize_train_set()
    test_ratio = 0.02
    pruning_samples = int(x_test.shape[0] * test_ratio)
    print("number of pruning samples" , pruning_samples)
    #ResNet_0.26_0.74_SGD.h5,0.565,mean_absolute_error
    temp = []
    #models = [pruning_folder + 'ResNet_0.26_0.74_SGD.h5']
    models = file_browser(pruning_folder)
    print("number of models in the pool", len(models))
    i = 0
    
    with open('acc_results.csv', 'w') as ff:

        pool_predictions = {}
        while i < len(models) :
            model_yhats = model_eval(models[i], x_test[:pruning_samples])
            model_acc = compute_acc2(x_test[:pruning_samples], y_test[:pruning_samples],models[i])
            #model_acc_manual = compute_accuracy(x_test[:pruning_samples], y_test[:pruning_samples],models[i])

            ff.write("%s, %s\n" % (models[i], model_acc))
            for one_sample_prediction in model_yhats:
                temp.append(np.argmax(one_sample_prediction))
            pool_predictions[i+1] = temp
            temp = []
            i += 1
    X = np.array(list(pool_predictions.values()))
    #kmeans_elbow_clustering(X)

    print(pool_predictions)
    clusters = kmeans_clustering(X)
    diversity_dic = {}
    for i in range (0,len(models)):
        diversity_dic[models[i]] = clusters.labels_[i]
    with open('clustering_results.csv', 'w') as f:
        for key in diversity_dic.keys():        
            f.write("%s, %s\n" % (key, diversity_dic[key]))

    


if __name__ == "__main__":
    main() 