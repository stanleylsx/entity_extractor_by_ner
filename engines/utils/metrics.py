# -*- coding: utf-8 -*-
# @Time : 2020/9/9 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from engines.utils.extract_entity import extract_entity


def metrics(X, y_true, y_pred, configs, data_manager):
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0

    label_num = {}
    label_metrics = {}
    measuring_metrics = configs.measuring_metrics
    # tensor向量不能直接索引，需要转成numpy
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    X = X.numpy()
    for i in range(len(y_true)):
        if configs.use_pretrained_model:
            x = data_manager.tokenizer.convert_ids_to_tokens(X[i].tolist(), skip_special_tokens=True)
        else:
            x = [str(data_manager.id2token[val]) for val in X[i] if val != data_manager.token2id[data_manager.PADDING]]
        y = [str(data_manager.id2label[val]) for val in y_true[i] if val != data_manager.label2id[data_manager.PADDING]]
        y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                 val != data_manager.label2id[data_manager.PADDING]]  # if val != 5

        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)

        true_labels, labeled_labels_true, _ = extract_entity(x, y, data_manager)
        pred_labels, labeled_labels_pred, _ = extract_entity(x, y_hat, data_manager)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))
        
        for label in data_manager.suffix:
            label_num.setdefault(label, {})
            label_num[label].setdefault('hit_num', 0)
            label_num[label].setdefault('pred_num', 0)
            label_num[label].setdefault('true_num', 0)

            true_lab = [x for (x, y) in zip(true_labels, labeled_labels_true) if y == label]
            pred_lab = [x for (x, y) in zip(pred_labels, labeled_labels_pred) if y == label]

            label_num[label]['hit_num'] += len(set(true_lab) & set(pred_lab))
            label_num[label]['pred_num'] += len(set(pred_lab))
            label_num[label]['true_num'] += len(set(true_lab))

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    # 按照字段切分
    for label in label_num.keys():
        tmp_precision = 0.0
        tmp_recall = 0.0
        tmp_f1 = 0.0
        # 只包括BI
        if label_num[label]['pred_num'] != 0:
            tmp_precision = 1.0 * label_num[label]['hit_num'] / label_num[label]['pred_num']
        if label_num[label]['true_num'] != 0:
            tmp_recall = 1.0 * label_num[label]['hit_num'] / label_num[label]['true_num']
        if tmp_precision > 0 and tmp_recall > 0:
            tmp_f1 = 2.0 * (tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
        label_metrics.setdefault(label, {})
        label_metrics[label]['precision'] = tmp_precision
        label_metrics[label]['recall'] = tmp_recall
        label_metrics[label]['f1'] = tmp_f1

    results = {}
    for measure in measuring_metrics:
        results[measure] = vars()[measure]
    return results, label_metrics
