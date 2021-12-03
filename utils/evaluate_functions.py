import numpy as np
import torch
from sklearn.metrics import ndcg_score


def compute_ndcg(test_data, user_feature_matrix, item_feature_matrix, k, model, device):
    model.eval()
    ndcgs = []
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            items = row[1]
            gt_labels = row[2]
            user_features = np.array([user_feature_matrix[user] for i in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            ndcg = ndcg_score([gt_labels], [scores], k=k)
            ndcgs.append(ndcg)
    ave_ndcg = np.mean(ndcgs)
    return ave_ndcg

def evaluate_user_perspective(user_perspective_data, u_i_expl_dict):
    pres = []
    recs = []
    f1s = []
    for u_i, gt_features in user_perspective_data.items():
        if u_i in u_i_expl_dict:
            TP = 0
            pre_features = u_i_expl_dict[u_i]
            # print('f: ', gt_features, pre_features)
            for feature in pre_features:
                if feature in gt_features:
                    TP += 1
            pre = TP / len(pre_features)
            rec = TP / len(gt_features)
            if (pre + rec) != 0:
                f1 = (2 * pre * rec) / (pre + rec)
            else:
                f1 = 0
            pres.append(pre)
            recs.append(rec)
            f1s.append(f1)
    ave_pre = np.mean(pres)
    ave_rec = np.mean(recs)
    ave_f1 = np.mean(f1s)
    return ave_pre, ave_rec, ave_f1


def evaluate_model_perspective(
        rec_dict,
        u_i_exp_dict,
        base_model,
        user_feature_matrix,
        item_feature_matrix,
        rec_k,
        device):
    """
    compute PN, PS and F_NS score for the explanations
    :param rec_dict: {u1: [i1, i2, i3, ...] , u2: [i1, i2, i3, ...]}
    :param u_i_exp_dict: {(u, i): [f1, f2, ...], ...}
    :param base_model: the trained base recommendation model
    :param user_feature_matrix: |u| x |p| matrix, the attention on each feature p for each user u
    :param item_feature_matrix: |i| x |p| matrix, the quality on each feature p for each item i
    :param rec_k: the length of the recommendation list, only generated explanations for the items on the list
    :param device: the device of the model
    :return: the mean of the PN, PS and FNS scores
    """
    pn_count = 0
    ps_count = 0
    for u_i, fs in u_i_exp_dict.items():
        user = u_i[0]
        target_item = u_i[1]
        features = set(fs)
        items = rec_dict[user]
        target_index = items.index(target_item)
        # compute PN
        cf_items_features = []
        for item in items:
            item_ori_feature = np.array(item_feature_matrix[item])
            item_cf_feature = np.array([0 if s in features else item_ori_feature[s]
                                        for s in range(len(item_ori_feature))], dtype='float32')
            cf_items_features.append(item_cf_feature)
        cf_ranking_scores = base_model(torch.from_numpy(np.array([user_feature_matrix[user]
                                                                      for i in range(len(cf_items_features))])
                                                            ).to(device),
                                           torch.from_numpy(np.array(cf_items_features)).to(device)).squeeze()
        cf_score_list = cf_ranking_scores.to('cpu').detach().numpy()
        sorted_index = np.argsort(cf_score_list)[::-1]
        cf_rank = np.argwhere(sorted_index == target_index)[0, 0]  # the updated ranking of the current item
        if cf_rank > rec_k - 1:
            pn_count += 1
        # compute NS
        cf_items_features = []
        for item in items:
            item_ori_feature = np.array(item_feature_matrix[item])
            item_cf_feature = np.array([item_ori_feature[s] if s in features else 0
                                        for s in range(len(item_ori_feature))], dtype='float32')
            cf_items_features.append(item_cf_feature)
        cf_ranking_scores = base_model(torch.from_numpy(np.array([user_feature_matrix[user]
                                                                      for i in range(len(cf_items_features))])
                                                            ).to(device),
                                           torch.from_numpy(np.array(cf_items_features)).to(device)).squeeze()
        cf_score_list = cf_ranking_scores.to('cpu').detach().numpy()
        sorted_index = np.argsort(cf_score_list)[::-1]
        cf_rank = np.argwhere(sorted_index == target_index)[0, 0]  # the updated ranking of the current item
        if cf_rank < rec_k:
            ps_count += 1
    if len(u_i_exp_dict) != 0:
        pn = pn_count / len(u_i_exp_dict)
        ps = ps_count / len(u_i_exp_dict)
        if (pn + ps) != 0:
            fns = (2 * pn * ps) / (pn + ps)
        else:
            fns = 0
    else:
        pn = 0
        ps = 0
        fns = 0
    return pn, ps, fns
