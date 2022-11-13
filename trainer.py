import copy
from sys import stderr
import time
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils import debug


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in tqdm(range(num_batches)):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            del graph
            del targets
            del predictions
            del batch_loss
        model.train()
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in tqdm(range(num_batches)):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            del graph
            del targets
            del predictions
            del batch_loss
        model.train()
        return np.mean(_loss).item(), \
                accuracy_score(all_targets, all_predictions) * 100, \
                precision_score(all_targets, all_predictions) * 100, \
                recall_score(all_targets, all_predictions) * 100, \
                f1_score(all_targets, all_predictions) * 100
    pass

def save_after_ggnn(model, num_batches, data_iter, file_name):
    model.eval()
    lst = []
    with torch.no_grad():
        for _ in range(num_batches):
            graph, targets = data_iter()
            output = model.output(graph, cuda=True)
            output = list(output.detach().cpu().numpy().astype(float)) # (128, 200)
            targets = targets.detach().cpu().numpy()
            for i in range(len(targets)):
                dic = {}
                dic['target'] = int(targets[i])
                dic['graph_feature'] = list(output[i])
                lst.append(dic)
    model.train()
    with open("output/" + file_name + ".json", "w+") as f:
        f.write(json.dumps(lst))

def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        for step_count in tqdm(range(max_steps)):
            model.train()
            model.zero_grad()
            # start_t = time.time()
            graph, targets = dataset.get_next_train_batch()
            # end_t = time.time()
            # print("get batch takes:", end_t - start_t)
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            # end_t_2 = time.time()
            # print("model forward takes:", end_t_2 - end_t)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            del graph
            del targets
            if step_count % dev_every == (dev_every - 1):
                valid_loss, valid_acc, valid_prec, valid_recall, valid_f1 = evaluate_metrics(model, loss_function, dataset.initialize_valid_batch(),
                                                     dataset.get_next_valid_batch)
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                debug('Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tAcc: %5.2f\tPrec: %5.2f\tRecall: %5.2f\tF1: %5.2f\tPatience %d' % (
                    step_count, np.mean(train_losses).item(), valid_loss, valid_acc, valid_prec, valid_recall, valid_f1, patience_counter))
                debug('=' * 100)
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)
