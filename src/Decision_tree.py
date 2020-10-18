import sklearn as sk

def train_using_gini(X_train, y_train):
       # Creating the classifier object
       clf_gini = sk.tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
       # Performing training
       clf_gini.fit(X_train, y_train)
       return clf_gini

def train_using_entropy(X_train, y_train):
       # Decision tree with entropy
       clf_entropy = sk.tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
       # Performing training
       clf_entropy.fit(X_train, y_train)
       return clf_entropy

def DecisionTreeclf(clf, data_feature_names):
       import pydotplus
       import collections
       dot_data = sk.tree.export_graphviz(clf, feature_names=data_feature_names, out_file=None, filled=True,
                                          rounded=True)
       graph = pydotplus.graph_from_dot_data(dot_data)

       colors = ('turquoise', 'orange')
       edges = collections.defaultdict(list)

       for edge in graph.get_edge_list():
              edges[edge.get_source()].append(int(edge.get_destination()))

       for edge in edges:
              edges[edge].sort()
              for i in range(2):
                     dest = graph.get_node(str(edges[edge][i]))[0]
                     dest.set_fillcolor(colors[i])

       graph.write_pdf('tree_entropy.pdf')