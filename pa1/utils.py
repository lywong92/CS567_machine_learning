import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: information gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    entr_children = 0
    samples_total = sum(sum(branches,[]))
    
    # calculate entropy for each child 
    for i in range(len(branches)):
        samples_per_row = sum(branches[i])
        entr_per_row =  0
        for j in range(len(branches[i])):
            ratio = branches[i][j] / samples_per_row
            if ratio > 0:
                entr_per_row -= ratio * np.log2(ratio) 
        entr_children += entr_per_row * (samples_per_row / samples_total)
    
    info_gain = S - entr_children
    return info_gain


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    
    # calculate error rate without pruning
    predicted = decisionTree.predict(X_test)
    mismatch = 0
    for i in range(len(y_test)):
        if predicted[i] != y_test[i]:
            mismatch += 1
    error_orig = mismatch / len(y_test)
    error_min = error_orig

    root = decisionTree.root_node
    nodes = [root]
    candidate_nodes = []
    prune_node = None
    
    # do a DFS on original tree and find all pruning candidate nodes
    while nodes:
        cur_node = nodes[0]
        if cur_node.children:
            candidate_nodes.append(cur_node)
        nodes = nodes[1:]
        for child in cur_node.children:
            nodes.insert(0, child)

    # find best improved error rate and node to prune
    for node in candidate_nodes:
        # temporarily prune candidate node to find updated error rate
        node.splittable = False
        new_predicted = decisionTree.predict(X_test)
        new_mismatch = 0
        for i in range(len(y_test)):
            if new_predicted[i] != y_test[i]:
                new_mismatch += 1
        error_cur = new_mismatch / len(y_test)
        
        if error_cur < error_min:
            error_min = error_cur
            prune_node = node
        # undo temporary pruning
        node.splittable = True
    
    # recursively prune until error rate doesn't improve
    if error_min < error_orig:        
        prune_node.children.clear()
        prune_node.splittable = False
        reduced_error_prunning(decisionTree, X_test, y_test)
        
    return

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])
    
    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


# TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    
    assert len(real_labels) == len(predicted_labels)
    # tp = true positive, fp = false positive, fn = false negative
    tp, fp, fn = 0, 0, 0
    for r, p in zip(real_labels, predicted_labels):
        if r == 1 and p == 1:
            tp += 1
        if r == 0 and p == 1:
            fp += 1
        if r == 1 and p == 0:
            fn += 1
            
    # f1 = 2*(precision*recall) / (precision+recall)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0 
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
    
# TODO: euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    temp = np.subtract(point1, point2)
    return np.sqrt(np.inner(temp, temp))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(point1, point2)


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    temp = np.subtract(point1, point2)
    return (-1)*np.exp((-0.5)*np.inner(temp, temp))


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    x_norm = np.linalg.norm(point1)
    y_norm = np.linalg.norm(point2)
    denom = x_norm*y_norm
    ans = 1-(np.inner(point1, point2)/denom) if denom != 0 else 0
    return ans


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    
    best_k, best_f1 = 0, -1
    
    for name, distance_func in distance_funcs.items():
        # train model with training data and find best k
        for k in range(1, 30, 2):
            model = KNN(k, distance_func)
            model.train(Xtrain, ytrain)
            valid_f1 = f1_score(yval, model.predict(Xval))
        
            if valid_f1 > best_f1:
                best_f1, best_k = valid_f1, k
                best_func, best_name = distance_func, name

    best_model = KNN(best_k, best_func)
    best_model.train(Xtrain, ytrain)
        
    return best_model, best_k, best_name


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    
    best_k, best_f1 = 0, -1
    
    for scaling_name, scaling_class in scaling_classes.items(): 
        for name, distance_func in distance_funcs.items():
            scaling_func = scaling_class()
            Xtrain_scaled = scaling_func(Xtrain)
            Xval_scaled = scaling_func(Xval)
            
            # train model with scaled training data and find best k
            for k in range(1, 30, 2):
                model = KNN(k, distance_func)
                model.train(Xtrain_scaled, ytrain)
                valid_f1 = f1_score(yval, model.predict(Xval_scaled))
            
                if valid_f1 > best_f1:
                    best_f1, best_k = valid_f1, k
                    best_func, best_name = distance_func, name
                    best_scaling_class = scaling_class
                    best_scaling_name = scaling_name
                
    best_model = KNN(best_k, best_func)
    best_scaling_func = best_scaling_class()
    best_model.train(best_scaling_func(Xtrain), ytrain)
        
    return best_model, best_k, best_name, best_scaling_name
    
    
class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        result = []
        for i in range(len(features)):
            norm = []
            denom = np.sqrt(np.inner(features[i], features[i]))
            for j in range(len(features[i])):
                if denom != 0:
                    norm.append(features[i][j]/denom)
                else:
                    norm.append(0)
            result.append(norm)
        return result


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.called = False
        self.data_max = []
        self.data_min = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        # store initial max and min information
        if not self.called:
            self.called = True
            data = np.reshape(features, (len(features),len(features[0])))
            self.data_max.extend(np.amax(data, axis=0))
            self.data_min.extend(np.amin(data, axis=0))
        
        # scale according to max and min of each column
        result = features.copy()
        for j in range(len(features[0])):
            for i in range(len(features)):
                denom = self.data_max[j]-self.data_min[j]
                if denom != 0:
                    result[i][j] = (result[i][j]-self.data_min[j])/denom
                else:
                    result[i][j] = 0.0     
        return result
