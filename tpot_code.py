import os
import sys
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import json


class DataLoader:
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir

    def load_labels(self, dataset_name, split):
        label_file = os.path.join(self.datasets_dir, f'{dataset_name}_{split}.label')
        if os.path.exists(label_file):
            labels = np.loadtxt(label_file, delimiter=',')
            return labels
        else:
            raise FileNotFoundError(f"Label file for {dataset_name} {split} split not found.")

    def load_text_embeddings(self, dataset_name, split):
        text_file = os.path.join(self.datasets_dir, f'{dataset_name}_{split}.txt')
        if os.path.exists(text_file):
            text_embeddings = np.loadtxt(text_file, delimiter=',')
            return text_embeddings
        else:
            raise FileNotFoundError(f"Text embeddings file for {dataset_name} {split} split not found.")

    def load_kg_embeddings(self, dataset_name, split):
        kg_file = os.path.join(self.datasets_dir, f'{dataset_name}_{split}.kg')
        if os.path.exists(kg_file):
            kg_embeddings = np.loadtxt(kg_file, delimiter=',')
            return kg_embeddings
        else:
            raise FileNotFoundError(f"KG embeddings file for {dataset_name} {split} split not found.")

    def load_dataset(self, dataset_name, split):
        labels = self.load_labels(dataset_name, split)
        text_embeddings = self.load_text_embeddings(dataset_name, split)
        kg_embeddings = self.load_kg_embeddings(dataset_name, split)
        return labels, text_embeddings, kg_embeddings

def concatenate_embeddings(text_embeddings, kg_embeddings):
    return np.hstack((text_embeddings, kg_embeddings))

def concatenate_train_valid(train, valid):
    return np.concatenate((train, valid), axis=0)

def project_embeddings(train_embeddings, test_embeddings, n_components):
    svd = TruncatedSVD(n_components=n_components)
    train_proj = svd.fit_transform(train_embeddings)
    test_proj = svd.transform(test_embeddings)
    return train_proj, test_proj

def run_tpot(data_folder, dataset_name, mode = 0, proj = 0):
    loader = DataLoader(data_folder)

    labels_train, text_embeddings_train, kg_embeddings_train = loader.load_dataset(dataset_name, 'train')
    labels_valid, text_embeddings_valid, kg_embeddings_valid = loader.load_dataset(dataset_name, 'valid')
    labels_test, text_embeddings_test, kg_embeddings_test = loader.load_dataset(dataset_name, 'test')

    
    labels_train = concatenate_train_valid(labels_train, labels_valid)
    text_embeddings_train = concatenate_train_valid(text_embeddings_train, text_embeddings_valid)
    kg_embeddings_train = concatenate_train_valid(kg_embeddings_train, kg_embeddings_valid)
    
    if mode == 0:
        X_train = np.hstack((text_embeddings_train, kg_embeddings_train))
        X_test = np.hstack((text_embeddings_test, kg_embeddings_test))
    elif mode == 1:
        X_train = text_embeddings_train
        X_test = text_embeddings_test
    elif mode == 2:
        X_train = kg_embeddings_train
        X_test = kg_embeddings_test
    if proj != 0:
        X_train, X_test = project_embeddings(X_train, X_test, proj)
    
    y_train = labels_train
    y_test = labels_test

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    tpot = TPOTClassifier(generations=100, population_size=100, early_stop = 5, verbosity=2, random_state=42, n_jobs=-1, max_time_mins  = 60)
    tpot.fit(X_train, y_train)

    score = tpot.score(X_test, y_test)
    preds = tpot.predict(X_test)
    data_folder = data_folder.replace('export','').replace('/','-')
    mmode = {0: 'concat', 1: 'text', 2: 'kg'}

    with open(f'models/{data_folder}_{dataset_name}_{mmode[mode]}_{proj}.json', 'w') as f:
        res_dict = {'predicitons' : preds.tolist(), 'dataset': dataset_name, 'true': y_test.tolist(), 'embedding': data_folder, 'mode': mmode[mode], 'dim': proj}
        json.dump(res_dict, f) 
    print(f'Accuracy for dataset {dataset_name}: {score}')
    tpot.export(os.path.join('results', f'{data_folder}_{dataset_name}_{mmode[mode]}_{proj}.py')) 



if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python run_tpot.py <data_folder> <dataset_name> <mode> <proj>")
        sys.exit(1)
    data_folder = sys.argv[1]
    dataset_name = sys.argv[2]
    mode = int(sys.argv[3])
    proj = int(sys.argv[4])
    run_tpot(data_folder, dataset_name, mode, proj)