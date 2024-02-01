import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    
    
    
     
    plt.scatter(X[y==-1,0], X[y==-1,1], c="green", alpha=0.5)
    plt.scatter(X[y==1,0], X[y==1,1], c="blue", alpha=0.5)
    plt.legend(['0','1'],loc="lower right", title="Classes")
    plt.xlabel("Symmetry_Feature")
    plt.ylabel("Intensity_Feature")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    #plt.show()
    plt.savefig("../train_features.pdf")


    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.scatter(X[y==-1,0], X[y==-1,1], c="green", alpha=0.5)
    plt.scatter(X[y==1,0], X[y==1,1], c="blue", alpha=0.5)
    plt.legend(['0','1'],loc="lower right", title="Classes")
    plt.xlabel("Symmetry_Feature")
    plt.ylabel("Intensity_Feature")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    
    x_para = np.array([X[:,0].min(), X[:,0].max()])
    m = -W[1]/W[2]
    c = -W[0]/W[2]
    decision_boundary = m*x_para + c
    plt.plot(x_para,decision_boundary,'--k')
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    #plt.show()
    plt.savefig("../train_result_sigmoid.pdf")
    
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
      
    plt.clf()
    plt.plot(X[y==0,0],X[y==0,1],'og',markersize=3)
    plt.plot(X[y==1,0],X[y==1,1],'ob',markersize=3)
    plt.plot(X[y==2,0],X[y==2,1],'or',markersize=3)
    plt.legend(['0','1','2'],loc="lower right", title="Classes")
    plt.xlabel("Symmetry_Feature")
    plt.ylabel("Intensity_Feature")
    symmetry_range = np.linspace(X[:,0].min(), X[:,0].max())
    decision_boundary1 = np.zeros(symmetry_range.shape)
    decision_boundary2 = np.zeros(symmetry_range.shape)
    for ix,x1 in enumerate(symmetry_range):
        w0, w1, w2 = (W[0], W[1], W[2])
        decision_boundary1[ix] = np.max([((w1[0] - w0[0]) + (w1[1] - w0[1])*x1)/(w0[2] - w1[2]), ((w2[0] - w0[0]) + (w2[1] - w0[1])*x1)/(w0[2] - w2[2])])
        decision_boundary2[ix] = np.min([((w0[0] - w1[0]) + (w0[1] - w1[1])*x1)/(w1[2] - w0[2]), ((w2[0] - w1[0]) + (w2[1] - w1[1])*x1)/(w1[2] - w2[2])])
    plt.plot(symmetry_range,decision_boundary1,'--k')
    plt.plot(symmetry_range,decision_boundary2,'--k')
    plt.ylim([-1,0])
    plt.xlim([-1,0])
    plt.savefig("../train_result_softmax.pdf")
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    best_LR = logistic_regression(learning_rate=0.1, max_iter=1200)
    best_LR.fit_miniBGD(train_X, train_y,25)
    print("\n\n Best Logistic Regression Sigmoid")
    print("Best Logistic Regression Sigmoid weights:", best_LR.get_params())
    print("Best Logistic Regression Sigmoid train accuracy: ",best_LR.score(train_X,train_y))
    print("Best Logistic Regression Sigmoid validation accuracy: ",best_LR.score(valid_X,valid_y))
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_LR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test, label_test = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test)  #process raw data for extracting features
    test_y_all, test_idx = prepare_y(label_test)  #process labels for all data  
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx] 
    test_y[np.where(test_y==2)] = -1
    data_shape= test_y.shape[0] 
    print("Test Accuracy:", best_LR.score(test_X, test_y))
    
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y,25)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    best_LR_multi = logistic_regression_multiclass(learning_rate=0.1, max_iter=1000,  k= 3)
    print("\n\n Best LR Multiclass")
    best_LR_multi.fit_miniBGD(train_X, train_y, 25)
    print("Best LR Multiclass weights:\n", best_LR_multi.get_params())
    print("Best LR Multiclass train accuracy: ",best_LR_multi.score(train_X, train_y))
    print("Best LR Multiclass validation accuracy: ",best_LR_multi.score(valid_X, valid_y))
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    visualize_result_multi(train_X[:, 1:3], train_y, best_LR_multi.get_params())
    print("Best LR Multiclass Test Accuracy: ", best_LR_multi.score(test_X_all, test_y_all))
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=1000,  k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("\n\n2-Class Softmax LR")
    print("Softmax LR weights:\n", logisticR_classifier_multiclass.get_params())
    print("Softmax LR train accuracy: ", logisticR_classifier_multiclass.score(train_X, train_y))
    print("Softmax LR validation accuracy: ", logisticR_classifier_multiclass.score(valid_X, valid_y))
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = 0
    print("Softmax LR test accuracy: ", logisticR_classifier_multiclass.score(test_X, test_y))
    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    best_LR = logistic_regression(learning_rate=0.02, max_iter=1000)
    best_LR.fit_miniBGD(train_X, train_y,10)
    print("\n\nBinary Sigmoid LR")
    print("Binary Sigmoid LR weights:\n", best_LR.get_params())
    print("Binary Sigmoid LR train accuracy: ",best_LR.score(train_X,train_y))
    print("Binary Sigmoid LR validation accuracy: ", best_LR.score(valid_X,valid_y))
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1
    print("Binary Sigmoid LR test accuracy: ", best_LR.score(test_X,test_y))
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


# '''
# Explore the training of these two classifiers and monitor the graidents/weights for each step. 
# Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
# Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
# '''
    ### YOUR CODE HERE
    sigmoid = logistic_regression(learning_rate=1, max_iter=1)
    softmax = logistic_regression_multiclass(learning_rate=0.5, max_iter=1,  k= 2)
    sigmoid.fit_miniBGD(train_X, train_y,10)
    train_y[np.where(train_y==-1)] = 0
    softmax.fit_miniBGD(train_X,train_y,10)
    print("sigmoid weights:\n", sigmoid.get_params())
    print("softmax weights:\n", softmax.get_params())
    ### END YOUR CODE

    # ------------End------------

if __name__ == '__main__':
    main()