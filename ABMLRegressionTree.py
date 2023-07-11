# Imports.
from collections import defaultdict
import random

import pandas as pd
import numpy as np
from sklearn.metrics import max_error, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import RepeatedKFold


metrics_map = [
    ("Max. err.", max_error),
    ("R2", r2_score),
    ("Explainer var.", explained_variance_score),
    ("MAE", mean_absolute_error),
    ("MSE", mean_squared_error),
    ("MAPE", mean_absolute_percentage_error),
]


class ABMLTreeNode():
    """
    Class representation of an ABML tree node.
    """
    def __init__(
        self,
        X: pd.DataFrame,
        Y: list,
        A: list,
        depth=0,
        node_type="split",
        rule=""
    ):
        self.X = X
        self.Y = Y
        self.A = A

        self.features = list(self.X.columns)

        self.depth = depth
        self.node_type = node_type
        self.rule = rule

        self.ymean = np.mean(Y)
        #self.residuals = self.Y - self.ymean
        #self.mse = self.calc_mse(Y, self.ymean)
        self.n = len(Y)

        self.left = None
        self.right = None

        self.split_feature = None
        self.split_value = None
    
    def print_subtree(self):
        """
        Prints the subtree
        """
        # Define the number of spaces
        pretext = self.depth * "|   " + "|---"
       
        print("{} {} (N={} Y={})".format(pretext, self.rule or "#", self.n, round(self.ymean, 3)))

        if self.left is not None:
            self.left.print_subtree()
        
        if self.right is not None:
            self.right.print_subtree()


class ABTree():
    """
    Class for creating an ABML regression tree.
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=5,
        arg_penalty=0,
        depth_penalty=0,
        enable_additive_arg_error=False,
    ):
        self.X = None
        self.Y = None
        self.A = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.arg_penalty = arg_penalty
        self.depth_penalty = depth_penalty
        self.enable_additive_arg_error= enable_additive_arg_error

        self.n = None
        self.features = None
        self.model = None

    @staticmethod
    def calc_mse(ytrue, ymean) -> float:
        """
        Calculate the mean squared error.
        """
        n = len(ytrue)
        r = ytrue - ymean
        r = r ** 2
        r = np.sum(r)

        return r / n

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list.
        """
        return np.convolve(x, np.ones(window), "valid") / window

    def best_split(self, X, Y, A, depth) -> tuple:
        """
        Given the X features and Y targets calculates the best split
        for a decision tree
        """
        # Creating a dataset for spliting
        df = X.copy()
        df["Y"] = Y
        df["A"] = A

        # Get all args in this node.
        curr_args = []
        for a in A:
            if a:
                curr_args.extend(a)

        # MSE for the base input
        mse_base = self.calc_mse(Y, np.mean(Y)) # TODO how to set this? if no args regular, otherwise multiply
        mse_base = mse_base + mse_base * self.arg_penalty

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Getting the left and right ys
                left_y = Xdf[Xdf[feature]>value]["Y"].values
                right_y = Xdf[Xdf[feature]<=value]["Y"].values

                if len(left_y) > 0 and len(right_y) > 0:
                    # Getting the left and right residuals
                    res_left = left_y - np.mean(left_y)
                    res_right = right_y - np.mean(right_y)

                    # Concatenating the residuals
                    r = np.concatenate((res_left, res_right), axis=None)

                    # Calculating the mse
                    mse_split = self.calc_mse(r, 0)

                    # If the split does not correspond to arguments, multiply with penalty
                    # Arguments are currently "anti-arguments", the split should not evaluate any of these as true
                    for a in curr_args:
                        if eval(a):
                            mse_split = mse_split + mse_split * self.arg_penalty + mse_split * self.depth_penalty * (1 / (depth + 1))
                            #print("Split not corresponding to arg.", a, feature, value)

                            if not self.enable_additive_arg_error:
                                break

                    # Checking if this is the best split so far
                    if mse_split < mse_base:
                        best_feature = feature
                        best_value = value

                        # Setting the best gain to the current one
                        mse_base = mse_split

        return (best_feature, best_value)

    def _fit(self, X, Y, A, depth=0):
        """
        Recursive method to create the ABML regression tree
        """
        # Making a df from the data
        df = X.copy()
        df["Y"] = Y
        df["A"] = A

        n = len(Y)

        curr_node = ABMLTreeNode(X, Y, A, depth=depth, node_type="leaf")

        if (depth >= self.max_depth) or (n < self.min_samples_split) or (n <= 1):
            return curr_node
        
        best_feature, best_value = self.best_split(X, Y, A, depth)

        if best_feature is None:
            return curr_node

        # Create the left and right node
        left_df, right_df = df[df[best_feature]>best_value].copy(), df[df[best_feature]<=best_value].copy()
        curr_node.left = self._fit(left_df[self.features], left_df["Y"].values.tolist(), left_df["A"].values.tolist(), depth=depth+1)
        curr_node.right = self._fit(right_df[self.features], right_df["Y"].values.tolist(), right_df["A"].values.tolist(), depth=depth+1)

        curr_node.node_type = "split"
        curr_node.rule = "{} > {}".format(best_feature, round(best_value, 3))
        curr_node.split_feature = best_feature
        curr_node.split_value = best_value

        return curr_node
    
    def fit(self, X: pd.DataFrame, Y: list, A: list):
        self.X = X
        self.Y = Y
        self.A = A

        self.n = len(Y)
        self.features = list(self.X.columns)

        self.model = self._fit(X, Y, A, 0)

    def predict(self, xi):
        curr_node = self.model
        
        while curr_node.node_type != "leaf":
            ftr, val = curr_node.split_feature, curr_node.split_value
            if xi[ftr] > val:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        
        return curr_node.ymean

    def predict_all(self, X):
        return X.apply(self.predict, axis="columns")
    
    def evaluate(self, p, Y):
        return { name: f(Y, p) for name, f in metrics_map }
    
    def cross_evaluate(self, X, Y, A, folds=5, n=5, always_use_AE=False):
        scores = []

        rkf = RepeatedKFold(n_splits=folds, n_repeats=n, random_state=2652124)
        for train_index, test_index in rkf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
            A_train, A_test = [A[i] for i in train_index], [A[i] for i in test_index]
            
            self.fit(X_train, Y_train, A_train)
            p = self.predict_all(X_test)

            scores.append(self.evaluate(p, Y_test))
            
        for name, _ in metrics_map:
            print("{}: {}".format(name, np.mean([score[name] for score in scores])))

    def get_critical_info(self, xi):
        curr_node = self.model
        path = ""
        leaf_samples = None
        
        while curr_node.node_type != "leaf":
            path = path + curr_node.rule + "\n"
            ftr, val = curr_node.split_feature, curr_node.split_value
            leaf_samples = (curr_node.X, curr_node.Y, curr_node.A)
            if xi[ftr] > val:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        
        return path, leaf_samples

    def get_critical_sample(self, X, Y, A, folds=5, n=5, always_use_AE=False):
        scores = np.zeros(len(Y))

        rkf = RepeatedKFold(n_splits=folds, n_repeats=n, random_state=2652124)
        for train_index, test_index in rkf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
            A_train, A_test = [A[i] for i in train_index], [A[i] for i in test_index]
            
            self.fit(X_train, Y_train, A_train)
            p = self.predict_all(X_test)

            scores[test_index] = scores[test_index] + (np.array(Y)[test_index] - p) ** 2

        scores = scores / n
        max_locs = np.where(scores == np.amax(scores))[0]

        print("\nTop errors:", sorted(scores)[-5:])
        print("\nCritical sample(s):")
        for max_loc in max_locs:
            print("i={}, X={}, Y={}".format(max_loc, X.iloc[max_loc], Y[max_loc]))
            path, leaf_samples = self.get_critical_info(X.iloc[max_loc]) # TODO: this should be averaged across all models
            print("\nPath:\n")
            print(path)
            
            leaf_X = leaf_samples[0]
            leaf_X["Y"] = leaf_samples[1]
            leaf_X["args"] = leaf_samples[2]
            print("Leaf samples:\n")
            print(leaf_X)

            df = X.copy()
            df["Y"] = Y
            print("\nDataset description:\n")
            print(df.describe())

            Y_desc = { "min": np.min(Y), "max": np.max(Y) }
            Yi_norm = (Y[max_loc] - Y_desc["min"]) / (Y_desc["max"] - Y_desc["min"])
            print()
            if Yi_norm > 0.7:
                print("Y is high. Description:")
            elif Yi_norm > 0.3:
                print("Y is medium. Description:")
            else:
                print("Y is low. Description:")
            
            print(Y_desc)
            print("Yi: {} | normalized: {}".format(Y[max_loc], Yi_norm))
            # TODO: add for other attributes
            

    def print(self):
        """
        Prints the ABML tree
        """
        if self.model is not None:
            self.model.print_subtree()
        else:
            print("Please fit the model first.")


def parse_arguments(args_ABML):
    parsed_args = []
    for arg in args_ABML:
        if arg is not np.nan:
            arg_array = []
            for arg_part in arg.split("&&"):
                if "<" in arg_part:
                    arg_tokens = arg_part.strip().split("<")
                    arg_code = "feature == \"{}\" and value >= {}".format(arg_tokens[0].strip(), arg_tokens[1].strip())
                    arg_array.append(arg_code)
                elif ">" in arg_part:
                    arg_tokens = arg_part.strip().split(">")
                    arg_code = "feature == \"{}\" and value <= {}".format(arg_tokens[0].strip(), arg_tokens[1].strip())
                    arg_array.append(arg_code)
                else:
                    print("Arg parsing error.")
            parsed_args.append(arg_array)
        else:
            parsed_args.append("")
    
    return parsed_args

if __name__ == "__main__":
    # Load and preprocess data.
    data = pd.read_csv("datasets/auto-mpg.csv")

    data = data[data["horsepower"]!="?"]

    #args_ABML = parse_arguments(data["ABMLARGS"])

    features = ["horsepower", "weight", "acceleration"]
    for ft in features:
        data[ft] = pd.to_numeric(data[ft])

    X = data[features]
    Y = data["mpg"].values.tolist()

    args_ABML = parse_arguments(generate_random_arguments(X, features))

    tree = ABTree(max_depth=3, min_samples_split=3, arg_penalty=0.5)
    #tree.fit(X, Y, args_ABML)
    #tree.print()
    
    #print(X.iloc[0])
    #print(Y[0])
    #print(tree.predict(X.iloc[0]))

    #tree.get_critical_sample(X, Y, args_ABML)
    tree.cross_evaluate(X, Y, args_ABML)