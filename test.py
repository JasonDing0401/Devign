from modules.model import GGNNSum
import json
import torch
from torch.nn import Linear, Sigmoid
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class simpleDataset(Dataset):
    def __init__(self, path):
        with open(path+"/test_GGNNinput_graph.json", "r") as f:
            dic_lst = json.load(f)
        self.lst = []
        for d in dic_lst:
            data = [d["graph_feature"], d["target"]]
            self.lst.append(data)
        del dic_lst
    def __len__(self):
        return len(self.lst)
    def __getitem__(self, idx):
        input, label = self.lst[idx]
        return {"input": input, "label": label}

test_dataset = simpleDataset("output/bugzilla_snykio_V3")
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = GGNNSum(input_dim=225, output_dim=225,
                        num_steps=10000, max_edge_types=16)
model.load_state_dict(torch.load(f"models/bugzilla_snykio_V3/GGNNSumModel-model.bin"))
# print(model.classifier.weight)
# classifier = Linear(in_features=225, out_features=1)
# classifier.load_state_dict(torch.load(f"models/bugzilla_snykio_V3/GGNNSumModel-model.bin"), strict=False)
# print(classifier.weight)
model.cuda()
model.eval()
actual_labels = []
pred_labels = []
with torch.no_grad():
    for i, data in enumerate(dataloader):
        input, label = data.values()
        input = torch.FloatTensor(input).cuda()
        output = model.classifier(input)
        output = Sigmoid()(output).squeeze(dim=-1)
        actual_labels.append(label.item())
        output = output.cpu().numpy()
        output = 0 if output <= 0.5 else 1
        pred_labels.append(output)
        if (i+1) % 5000 == 0:
            print(i+1)

TN, FP, FN, TP = confusion_matrix(actual_labels, pred_labels).ravel()
print(accuracy_score(actual_labels, pred_labels) * 100, \
                precision_score(actual_labels, pred_labels) * 100, \
                recall_score(actual_labels, pred_labels) * 100, \
                f1_score(actual_labels, pred_labels) * 100, \
                round(TN/(TN+FP), 4)*100, \
                round(FP/(FP+TN), 4)*100, \
                round(FN/(TP+FN), 4)*100, sep=',')