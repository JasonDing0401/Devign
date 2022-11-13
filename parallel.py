import json
import sys
import ijson
import os
import pickle
import argparse

import torch
from dgl import DGLGraph

location = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/'
with open(location + 'edge_types.json', 'r') as f:
    edge_types = json.load(f)
# print(edge_types)
class DataEntry:
    def __init__(self, num_nodes, features, edges, target):
        self.num_nodes = num_nodes
        self.target = target
        # self.graph = DGLGraph().to("cuda")
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = edge_types.get(str(_type))
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
file = args.file
type = file.split('_')[0]
print(f'Reading {type} file!')
dir = f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/{type}/'
store = f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/devign/{type}/'
with open(dir + file, 'r') as f:
    entry = json.load(f)
    example = DataEntry(num_nodes=len(entry['node_features']),
                features=entry['node_features'],
                edges=entry['graph'], target=entry['targets'][0][0])
    cnt = file.split('_')[-1][:-5]
    with open(store + f'{type}_GGNNinput_'+cnt+'.pkl', 'wb+') as f:
        pickle.dump(example, f)

def collect_json_names():
    f = open('/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/file_names.txt', 'a')
    print('Reading Train File!')
    train_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/train/'
    train_store = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/devign/train/'
    os.makedirs(train_store, exist_ok=True)
    for name in os.listdir(train_dir):
        if not name.endswith('.json'):
            continue
        f.write(name+'\n')

    print('Reading Valid File!')
    valid_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/valid/'
    valid_store = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/devign/valid/'
    os.makedirs(valid_store, exist_ok=True)
    for name in os.listdir(valid_dir):
        if not name.endswith('.json'):
            continue
        f.write(name+'\n')
            
    print('Reading Test File!')
    test_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/test/'
    test_store = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/devign/test/'
    os.makedirs(test_store, exist_ok=True)
    for name in os.listdir(test_dir):
        if not name.endswith('.json'):
            continue
        f.write(name+'\n')

def update_edge_types():
    with open('/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/edge_types.json', 'r') as f:
        edge_types = json.load(f)
    max_etype = len(edge_types)
    print('Reading Train File!')
    train_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/train/'
    for name in os.listdir(train_dir):
        if not name.endswith('.json'):
            continue
        with open(train_dir + name, 'r') as f:
            data = json.load(f)
            for _, _type, _ in data['graph']:
                if _type not in edge_types:
                    edge_types[_type] = max_etype
                    max_etype += 1
    print('Reading Validation File!')
    valid_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/valid/'
    for name in os.listdir(valid_dir):
        if not name.endswith('.json'):
            continue
        with open(valid_dir + name, 'r') as f:
            data = json.load(f)
            for _, _type, _ in data['graph']:
                if _type not in edge_types:
                    edge_types[_type] = max_etype
                    max_etype += 1
    print('Reading Test File!')
    test_dir = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/test/'
    for name in os.listdir(test_dir):
        if not name.endswith('.json'):
            continue
        with open(test_dir + name, 'r') as f:
            data = json.load(f)
            for _, _type, _ in data['graph']:
                if _type not in edge_types:
                    edge_types[_type] = max_etype
                    max_etype += 1
    print("max_etype:", max_etype)
    print('saving edge type dict')
    location = '/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/split/'
    with open(location + 'edge_types.json', 'w+') as f:
        json.dump(edge_types, f)