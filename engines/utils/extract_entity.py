import re


def extract_entity_(sentence, labels_, reg_str, label_level):
    entices = []
    labeled_labels = []
    labeled_indices = []
    labels__ = [('%03d' % ind) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = ' '.join(labels__)

    re_entity = re.compile(reg_str)

    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labeled_labels.append('_')
        elif label_level == 2:
            labeled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])
        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        entity = ' '.join(sentence[start_index:end_index])
        labels = labels__[end_index:]
        labels = ' '.join(labels)
        entices.append(entity)
        labeled_indices.append((start_index, end_index))
        m = re_entity.search(labels)

    return entices, labeled_labels, labeled_indices


def extract_entity(x, y, data_manager):
    label_scheme = data_manager.label_scheme
    label_level = data_manager.label_level
    label_hyphen = data_manager.hyphen
    reg_str = ''
    if label_scheme == 'BIO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r'\s*)*'

    elif label_scheme == 'BIESO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E' + label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S' + label_hyphen + tag_str + r' )'

    return extract_entity_(x, y, reg_str, label_level)
