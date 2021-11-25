import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(0)


def get_feature_list(sentiment_data):
    """
    from user sentiment data, get all the features [F1, F2, ..., Fk] mentioned in the reviews
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: feature set F
    """
    feature_list = []
    for row in sentiment_data:
        for fos in row[2:]:
            feature = fos[0]
            if feature not in feature_list:
                feature_list.append(feature)
    feature_list = np.array(feature_list)
    return feature_list


def get_user_attention_matrix(sentiment_data, user_num, feature_list, max_range=5):
    """
    build user attention matrix
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :param user_num: number of users
    :param feature_list: [F1, F2, ..., Fk]
    :param max_range: normalize the attention value to [1, max_range]
    :return: the user attention matrix, Xij is user i's attention on feature j
    """
    user_counting_matrix = np.zeros((user_num, len(feature_list)))  # tij = x if user i mention feature j x times
    for row in sentiment_data:
        user = row[0]
        for fos in row[2:]:
            feature = fos[0]
            user_counting_matrix[user, feature] += 1
    user_attention_matrix = np.zeros((user_num, len(feature_list)))  # xij = [1-N], normalized attention matrix
    for i in range(len(user_counting_matrix)):
        for j in range(len(user_counting_matrix[i])):
            if user_counting_matrix[i, j] == 0:
                norm_v = 0  # if nor mentioned: 0
            else:
                norm_v = 1 + (max_range - 1) * ((2 / (1 + np.exp(-user_counting_matrix[i, j]))) - 1)  # norm score
            user_attention_matrix[i, j] = norm_v
    user_attention_matrix = np.array(user_attention_matrix)
    return user_attention_matrix


def get_item_quality_matrix(sentiment_data, item_num, feature_list, max_range=5):
    """
    build item quality matrix
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :param item_num: number of items
    :param feature_list: [F1, F2, ..., Fk]
    :param max_range: normalize the quality value to [1, max_range]
    :return: the item quality matrix, Yij is item i's quality on feature j
    """
    item_counting_matrix = np.zeros((item_num, len(feature_list)))  # kij = x if item i's feature j is mentioned x times
    item_sentiment_matrix = np.zeros((item_num, len(feature_list)))  # sij = x if the overall rating is x (sum up)
    for row in sentiment_data:
        item = row[1]
        for fos in row[2:]:
            feature = fos[0]
            sentiment = fos[2]
            item_counting_matrix[item, feature] += 1
            if sentiment == '+1':
                item_sentiment_matrix[item, feature] += 1
            elif sentiment == '-1':
                item_sentiment_matrix[item, feature] -= 1
            else:
                print("sentiment data error: the sentiment value can only be +1 or -1")
                exit(1)
    item_quality_matrix = np.zeros((item_num, len(feature_list)))
    for i in range(len(item_counting_matrix)):
        for j in range(len(item_counting_matrix[i])):
            if item_counting_matrix[i, j] == 0:
                norm_v = 0  # if not mentioned: 0
            else:
                norm_v = 1 + ((max_range - 1) / (1 + np.exp(-item_sentiment_matrix[i, j])))  # norm score
            item_quality_matrix[i, j] = norm_v
    item_quality_matrix = np.array(item_quality_matrix)
    return item_quality_matrix


def get_user_item_dict(sentiment_data):
    """
    build user & item dictionary
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: user dictionary {u1:[i, i, i...], u2:[i, i, i...]}, similarly, item dictionary
    """
    user_dict = {}
    item_dict = {}
    for row in sentiment_data:
        user = row[0]
        item = row[1]
        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
        if item not in item_dict:
            item_dict[item] = [user]
        else:
            item_dict[item].append(user)
    return user_dict, item_dict


def get_user_item_set(sentiment_data):
    """
    get user item set
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: user_set = set(u1, u2, ..., um); item_set = (i1, i2, ..., in)
    """
    user_set = set()
    item_set = set()
    for row in sentiment_data:
        user = row[0]
        item = row[1]
        user_set.add(user)
        item_set.add(item)
    return user_set, item_set


def sample_training_evaluation_pairs(user, training_dict, item_set,
                                     eval_num=1, sample_ratio=10):
    positive_items = set(training_dict[user])
    negative_items = set()
    for item in item_set:
        if item not in positive_items:
            negative_items.add(item)
    neg_length = len(positive_items) * sample_ratio
    negative_items = np.random.choice(np.array(list(negative_items)), neg_length, replace=False)
    train_positive = list(positive_items)[:-eval_num]
    eval_positive = list(positive_items)[-eval_num:]
    train_negative = list(negative_items)[:-(eval_num * sample_ratio)]
    eval_negative = list(negative_items)[-(eval_num * sample_ratio):]
    train_pairs = []
    eval_pairs = []
    for p_item in train_positive:
        train_pairs.append([user, p_item, 1])
    for n_item in train_negative:
        train_pairs.append([user, n_item, 0])
    for p_item in eval_positive:
        eval_pairs.append([user, p_item, 1])
    for n_item in eval_negative:
        eval_pairs.append([user, n_item, 0])
    return train_pairs, eval_pairs


def check_string(string):
    # if the string contains letters
    string_lowercase = string.lower()
    contains_letters = string_lowercase.islower()
    return contains_letters


def visualization(train_losses, val_losses, path):
    plt.plot(np.arange(len(train_losses)), train_losses, label='training loss')
    plt.plot(np.array(len(val_losses)), val_losses, label='validation loss')
    plt.legend()
    plt.savefig(path)
    plt.clf()


def get_mask_vec(user_attantion, k):
    """
    get the top-k mask for features. The counterfactual explanations can only be chosen from this space
    :param user_attantion: user's attantion vector on all the features
    :param k: the k from mask
    :return: a mask vector with 1's on the top-k features that the user cares about and 0's for others.
    """
    top_indices = np.argsort(user_attantion)[::-1][:k]
    mask = [0 for i in range(len(user_attantion))]
    for index in top_indices:
        if user_attantion[index] > 0:  # only consider the user mentioned features
            mask[index] = 1
    return np.array(mask)


def feature_filtering(sentiment_data, valid_features):
    """
    filter the sentiment data, remove the invalid features
    :param sentiment_data: [userID, itemID, [fos triplet 1], [fos triplet 2], ...]
    :param valid_features: set of valid features
    :return: the filtered sentiment data
    """
    cleaned_sentiment_data = []
    for row in sentiment_data:
        user = row[0]
        item = row[1]
        cleaned_sentiment_data.append([user, item])
        for fos in row[2:]:
            if fos[0] in valid_features:
                cleaned_sentiment_data[-1].append(fos)
        if len(cleaned_sentiment_data[-1]) == 2:
            del cleaned_sentiment_data[-1]
    return np.array(cleaned_sentiment_data)