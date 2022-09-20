import warnings
import pandas as pd
from sklearn.svm import SVC
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def plot_images(X, Y, num_images = 6):
    """
    This function plots images
    """
    plt.figure(figsize=(10,6))
    for i in range(1, num_images+1):
        plt.subplot(int(num_images/3), 3, i)
        plt.imshow(X[i], cmap='gray')
        plt.title("Digit " + str(Y[i]))
        plt.axis('off')
    plt.show()

def data_splitting(X, Y):
    """
    This function splits the data into train, dev, and test set.
    """
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, train_size = 0.92, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size = 0.96, random_state=42)
    return (X_train,Y_train), (X_dev, Y_dev), (X_test, Y_test)
    
    
def hyperparameter_tuning(parameter_list, xtrain, ytrain, xdev, ydev, xtest, ytest):
    """
    This function performs an extensive hyperparameter tuning
    for a given estimator with parameters list.
    """
    ## dataframe to store scores
    df_score = pd.DataFrame(columns = ['combination', 'train_acc', 'dev_acc'])
    
    ## get combination
    keys, values = zip(*parameter_list.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    ## training model for all combination
    for combination in tqdm(combinations):
        estimator = SVC(**combination)
        estimator.fit(xtrain, ytrain)
        train = estimator.score(xtrain,ytrain)
        dev = estimator.score(xdev,ydev)
        
        df_score = df_score.append({'combination' : combination, 'train_acc' : train, 'dev_acc' : dev},
                                   ignore_index=True)
    ## preprocessing
    df_score.index = df_score['combination']
    df_score = df_score.iloc[:, 1:]
    print(df_score)
    
    best_combination = df_score[df_score['dev_acc'] == df_score['dev_acc'].max()].index[0]
    
    ## training on best model
    best_estimator = SVC(**best_combination)
    best_estimator.fit(xtrain, ytrain)
    
    train = best_estimator.score(xtrain,ytrain)
    dev   = best_estimator.score(xdev, ydev)
    test  = best_estimator.score(xtest, ytest)
    
    print("\nThe best combination and Score...............................")
    final_score = pd.DataFrame({'gamma' : [best_combination['gamma']], 
                                'C' : [best_combination['C']], 
                                'Train Acc' : [round(train, 4)],
                                'Dev Acc' : [round(dev, 4)],
                                'Test Acc' : [round(test, 4)]})
    print(final_score)
    return best_estimator

if __name__ == '__main__':
    
    parameter_list = {'gamma' : ['auto', 'scale'],
                      'C' : [0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 500, 1000]}
    
    X_image          = load_digits()['images']
    X_image_reshaped = load_digits()['data']
    Y_image          = load_digits()['target']
    inp_len = len(X_image)
    
    print("Training On Original Images..............................................")
    print(f"Image Size : {X_image[0].shape}")
    plot_images(X_image, Y_image)
    (X_train,Y_train), (X_dev, Y_dev), (X_test, Y_test) = data_splitting(X_image_reshaped, Y_image)
    best_estimator_orig = hyperparameter_tuning(parameter_list, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
    
    
    
    print("\n\nTraining On Resized Images 1..............................................")
    X_image_resl1 = resize(X_image, (inp_len, 50, 50))
    X_image_resl1_reshaped = X_image_resl1.reshape(inp_len, 50*50)
    print(f"Image Size Resolution 1 : {X_image_resl1[0].shape}")
    plot_images(X_image_resl1, Y_image)
    (X_train1,Y_train1), (X_dev1, Y_dev1), (X_test1, Y_test1) = data_splitting(X_image_resl1_reshaped, Y_image)
    best_estimator_resl1 = hyperparameter_tuning(parameter_list, X_train1, Y_train1,
                                                 X_dev1, Y_dev1, X_test1, Y_test1)
    
    
    print("\n\nTraining On Resized Images 2..............................................")
    X_image_resl2 = resize(X_image, (inp_len, 32, 40))
    X_image_resl2_reshaped = X_image_resl2.reshape(inp_len, 32*40)
    print(f"Image Size Resolution 2 : {X_image_resl2[0].shape}")
    plot_images(X_image_resl2, Y_image)
    (X_train2,Y_train2), (X_dev2, Y_dev2), (X_test2, Y_test2) = data_splitting(X_image_resl2_reshaped, Y_image)
    best_estimator_resl2 = hyperparameter_tuning(parameter_list, X_train2, Y_train2,
                                                 X_dev2, Y_dev2, X_test2, Y_test2)
    
    print("\n\nTraining On Resized Images 3..............................................")
    X_image_resl3 = resize(X_image, (inp_len, 32, 16))
    X_image_resl3_reshaped = X_image_resl3.reshape(inp_len, 32*16)
    print(f"Image Size Resolution 3 : {X_image_resl3[0].shape}")
    plot_images(X_image_resl3, Y_image)
    (X_train3,Y_train3), (X_dev3, Y_dev3), (X_test3, Y_test3) = data_splitting(X_image_resl3_reshaped, Y_image)
    best_estimator_resl3 = hyperparameter_tuning(parameter_list, X_train3, Y_train3,
                                                 X_dev3, Y_dev3, X_test3, Y_test3)