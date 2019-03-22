import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
            
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
            
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) == 1:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split
        
        self.attr_used = []
        
        self.parent = None

    #TODO: try to split current node
    def split(self):
        
        if self.splittable == True:
            
            entropy_p, info_gain_max = 0,0
            attr_size_best = -1
            
            class_values = np.unique(self.labels)
            # calculate parent entropy
            for value_uniq in class_values:
                ratio = self.labels.count(value_uniq) / len(self.labels)
                entropy_p -= ratio * np.log2(ratio)
            
            # construct branch matrix to pass into info gain function
            # for each attribute, find unique values
            # unique labels already generated above (class_values)
            # create branch matrix with dimension of (# unique values)*(# unique labels)
            attr_all = [i for i in range(len(self.features[0]))]
            attr_available = [a for a in attr_all if a not in self.attr_used]
            for j in attr_available:   
                features_arr = np.array(self.features)
                attr_values = np.unique(features_arr[:,j])
                branch = np.zeros((attr_values.size, class_values.size)).tolist()
                for i in range(len(self.features)):
                    # populate branch matrix with counts 
                    for x in range(len(attr_values)):
                        for y in range(len(class_values)):
                            if (self.features[i][j] == attr_values[x] 
                                and self.labels[i] == class_values[y]):
                                branch[x][y] += 1
            
                # find info gain if split on current attribute
                info_gain = Util.Information_Gain(entropy_p, branch)
                # store best attribute used to split
                # break ties
                if (info_gain > info_gain_max or (info_gain == info_gain_max 
                    and len(attr_values) > attr_size_best)):
                    info_gain_max = info_gain
                    self.dim_split = j
                    self.feature_uniq_split = attr_values.tolist()
                    attr_size_best = len(attr_values)
            
            # create child nodes
            for value in self.feature_uniq_split:
                features_child = []
                labels_child = []
                value_indices_arr = np.array(self.features)
                value_indices = np.where(value_indices_arr[:,self.dim_split] == value)[0]
                for index in value_indices:
                    features_child.append(self.features[index])
                    labels_child.append(self.labels[index])
                num_cls_child = np.unique(labels_child).size
                
                child = TreeNode(features_child, labels_child, num_cls_child)
                child.attr_used.extend(self.attr_used)
                child.attr_used.append(self.dim_split)
                if len(child.attr_used) == len(self.features[0]):
                    child.splittable = False
                self.children.append(child)

            # recursively call split to create rest of tree
            for child in self.children:
                if child.splittable == True:
                    child.split()

        return
                    

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        # if attribute value can be found in current node's children, traverse down the tree
        if self.splittable == True and feature[self.dim_split] in self.feature_uniq_split:
            index = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[index].predict(feature)
        else:
            return self.cls_max
            
        
