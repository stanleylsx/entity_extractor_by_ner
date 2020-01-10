from engines.utils.ExtractEntity import extract_entity


def metrics(X, y_true, y_pred, measuring_metrics, data_manager):
    precision = -1.0
    recall = -1.0
    f1 = -1.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0
    for i in range(len(y_true)):
        x = [str(data_manager.id2token[val]) for val in X[i] if val != data_manager.token2id[data_manager.PADDING]]
        y = [str(data_manager.id2label[val]) for val in y_true[i] if val != data_manager.label2id[data_manager.PADDING]]
        y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                 val != data_manager.label2id[data_manager.PADDING]]  # if val != 5

        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)

        true_labels, labeled_labels, _ = extract_entity(x, y, data_manager)
        pred_labels, labeled_labels, _ = extract_entity(x, y_hat, data_manager)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    results = {}
    for measure in measuring_metrics:
        results[measure] = vars()[measure]
    return results
