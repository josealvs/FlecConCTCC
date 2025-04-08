import argparse
import os
import warnings
from random import seed

from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression as reglog

from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon
from src.utils import calculate_mean_stdev, list_knn_full, list_knn_Prelax, \
    list_knn_seeds, list_tree, result, select_labels

crs = [0.05]
thresholds = [0.95]

warnings.simplefilter("ignore")

parent_dir = "path_for_results"
datasets_dir = "./datasets"
datasets = sorted(os.listdir(datasets_dir))
init_labelled = [0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.23, 0.25]

# Lista de classificadores para rodar automaticamente
classifiers = [1, 2, 3, 4, 5, 6, 7]  # Modificado: agora percorre todos automaticamente

for classifier in classifiers:  # Modificado: Loop para rodar todos os classificadores
    fold_result_acc_final = []
    fold_result_f1_score_final = []

    comite_map = {
        1: "Comite_Naive_",
        2: "Comite_Tree_",
        3: "Comite_KNN_",
        5: "Comite_RandomForest_",
        6: "Comite_SVM_",
        7: "Comite_RegLog_"
    }

    comite = comite_map.get(classifier, "Comite_Heterogeneo_")
    
    folder_check_csv = parent_dir
    os.makedirs(folder_check_csv, exist_ok=True)

    file_check = f'{comite}.csv'
    check = os.path.join(folder_check_csv, file_check)

    if not os.path.exists(check):
        with open(check, 'a') as f:
            f.write('"ROUNDS", "DATASET","LABELLED-LEVEL","ACC","F1-SCORE"')

    file_check = f'{comite}F.csv'
    check = os.path.join(folder_check_csv, file_check)

    if not os.path.exists(check):
        with open(check, 'a') as f:
            header = (
                '"DATASET","LABELLED-LEVEL","ACC-AVERAGE",'
                '"STANDARD-DEVIATION-ACC","F1-SCORE-AVERAGE",'
                '"STANDARD-DEVIATION-F1-SCORE"'
            )
            f.write(header)

    for threshold in thresholds:
        for cr in crs:
            for labelled_level in init_labelled:
                for dataset in datasets:
                    comite = Ensemble(SelfFlexCon, cr=cr, threshold=threshold)
                    fold_result_acc = []
                    fold_result_f1_score = []
                    df = read_csv(f'datasets/{dataset}', header=0)
                    seed(214)
                    kfold = StratifiedKFold(n_splits=10)

                    _instances = df.iloc[:, :-1].values  # X
                    _target_unlabelled = df.iloc[:, -1].values  # Y

                    rounds = 0
                    for train, test in kfold.split(_instances, _target_unlabelled):
                        X_train = _instances[train]
                        X_test = _instances[test]
                        y_train = _target_unlabelled[train]
                        y_test = _target_unlabelled[test]
                        labelled_instances = labelled_level

                        rounds += 1
                        y = select_labels(y_train, X_train, labelled_instances)

                        if classifier == 1 or classifier == 4:
                            for i in range(9):
                                comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))

                        if classifier == 2 or classifier == 4:
                            for i in list_tree:
                                comite.add_classifier(i)

                        if classifier == 3 or classifier == 4:
                            if dataset == 'Seeds.csv':
                                for i in list_knn_seeds:
                                    comite.add_classifier(i)
                            elif dataset == 'PlanningRelax.csv':
                                for i in list_knn_Prelax:
                                    comite.add_classifier(i)
                            else:
                                for i in list_knn_full:
                                    comite.add_classifier(i)

                        if classifier == 5 or classifier == 4:
                            for i in range(9):
                                comite.add_classifier(RandomForestClassifier(n_estimators=1 * (i + 1), random_state=42))

                        if classifier == 6 or classifier == 4:
                            kernels = ['linear', 'rbf', 'poly']
                            for kernel in kernels:
                                comite.add_classifier(svm(kernel=kernel, C=1.0, probability=True, random_state=42))

                        if classifier == 7 or classifier == 4:
                            solvers = ['liblinear', 'lbfgs']
                            for solver in solvers:
                                comite.add_classifier(reglog(solver=solver, max_iter=200, random_state=42))

                        comite.fit_ensemble(X_train, y)
                        y_pred = comite.predict(X_test)
                        result_acc = accuracy_score(y_test, y_pred)

                        fold_result_acc.append(result_acc)
                        fold_result_acc_final.append(result_acc)

                        result_f1 = result(
                            classifier,
                            dataset,
                            y_test,
                            y_pred,
                            parent_dir,
                            labelled_level,
                            rounds
                        )

                        fold_result_f1_score.append(result_f1)
                        fold_result_f1_score_final.append(result_f1)

                    calculate_mean_stdev(
                        fold_result_acc,
                        classifier,
                        labelled_level,
                        parent_dir,
                        dataset,
                        fold_result_f1_score
                    )

    calculate_mean_stdev(
        fold_result_acc_final,
        classifier,
        labelled_level,
        parent_dir,
        'FINAL-RESULTS',
        fold_result_f1_score_final
    )
