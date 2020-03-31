import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
from mfcc_coeff import extract_features
import time
import warnings
warnings.filterwarnings("ignore")

source   = "samples\\"   

train_list = "training_sample_list.txt"  
test_list = "testing_sample_list.txt"      
file_paths = open(train_list,'r')

models = []
speakers = []
count = 1
# Extracting features for each speaker (3 files per speakers)
features = np.asarray(())
for path in file_paths:
    path = path.strip() 
    
    # read the wav file
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features from mfcc_coeff.py
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 3 files of speaker are concatenated, then do model training
    # here, we have selected GMM components to be 16.
    if count == 3:    
        gmm = GaussianMixture(n_components = 32, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # saving the gmm model of a person
        models.append(gmm)
        speakers.append(path.split("\\")[0].split("-")[0])
        features = np.asarray(())
        count = 0
        #print('modeled '+speakers[-1])
    count = count + 1
    
file_paths.close()    
file_paths = open(test_list,'r')

num_files = 0
correct_files = 0

for path in file_paths:   
    
    path = path.strip()   
    actual_speaker = path.split("\\")[0].split("-")[0]
    #print (path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print ("\tdetected as - ", speakers[winner])
    num_files += 1
    if actual_speaker == speakers[winner]:
        correct_files += 1
    time.sleep(1.0)


#seconds = time.time()
#print(seconds)
accuracy = correct_files * 100 / num_files
print("Accuracy = %s %%" % accuracy)