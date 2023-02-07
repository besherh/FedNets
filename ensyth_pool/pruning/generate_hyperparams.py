import random
import csv
import numpy as np

num_train_samples = 500
with open('constant_spar_hyperparams.csv','w') as csv_file:    
       filewriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
       filewriter.writerow(['epochs', 'batch_size','loss','opt','target_sparsity','frequency','model_id'])       
       for i in range (num_train_samples):
           epoch = random.choice( [2,3,4,5,6])
           batch_size = random.choice([16,32,64])
           loss = random.choice(['categorical_crossentropy','mean_squared_error','mean_absolute_error'])
           opt = random.choice(['adam'])
           target_sparsity = random.uniform(0.2,0.55)
           frequency = random.choice([50,75,100])
           raw = [epoch,batch_size,loss,opt,round(target_sparsity,2),frequency,i]
           filewriter.writerow(raw)

print("Done! Check your CSV file.")
