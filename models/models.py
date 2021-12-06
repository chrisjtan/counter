from numpy import core
import torch
import tqdm
import numpy as np

class BaseRecModel(torch.nn.Module):
    def __init__(self, feature_length):
        super(BaseRecModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_length * 2, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user_feature, item_feature):
        fusion = torch.cat((user_feature, item_feature), 1)
        out = self.fc(fusion)
        return out

class ExpOptimizationModel(torch.nn.Module):
    def __init__(self, base_model, rec_dataset, device, exp_args):
        super(ExpOptimizationModel, self).__init__()
        self.base_model = base_model
        self.rec_dataset = rec_dataset
        self.device = device
        self.exp_args = exp_args
        self.u_i_exp_dict = {}  # {(user, item): [f1, f2, f3 ...], ...}
        self.user_feature_matrix = torch.from_numpy(self.rec_dataset.user_feature_matrix).to(self.device)
        self.item_feature_matrix = torch.from_numpy(self.rec_dataset.item_feature_matrix).to(self.device)
        self.rec_dict, self.user_perspective_test_data = self.generate_rec_dict()

    def generate_rec_dict(self):
        rec_dict = {}
        correct_rec_dict = {}  # used for user-side evaluation
        for row in self.rec_dataset.test_data:
            user = row[0]
            items = row[1]
            labels = row[2]
            correct_rec_dict[user] = []
            user_features = self.user_feature_matrix[user].repeat(len(items), 1)
            scores = self.base_model(user_features,
                        self.item_feature_matrix[items]).squeeze()
            scores = np.array(scores.to('cpu'))
            sort_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            sorted_items = [items[i] for i in sort_index]
            rec_dict[user] = sorted_items
            for i in range(self.exp_args.rec_k):  # find the correct items and add to the user side test data
                if labels[sort_index[i]] == 1:
                    correct_rec_dict[user].append(items[sort_index[i]])
        user_item_feature_dict = {}  # {(u, i): f, (u, i): f]

        for row in self.rec_dataset.sentiment_data:
            user = row[0]
            item = row[1]
            user_item_feature_dict[(user, item)] = []
            for fos in row[2:]:
                feature = fos[0]
                user_item_feature_dict[(user, item)].append(feature)
        user_perspective_test_data = {}  # {(u, i):f, (u, i): f]}
        for user, tiems in correct_rec_dict.items():
            for item in tiems:
                feature = user_item_feature_dict[(user, item)]
                user_perspective_test_data[(user, item)] = feature
        return rec_dict, user_perspective_test_data

    def generate_explanation(self):
        # u_i_exps_dict = {}  # {(user, item): [f1, f2, f3 ...], ...}
        exp_nums = []
        exp_complexities = []
        no_exp_count = 0
        if self.exp_args.test_num == -1:
            test_num = len(list(self.rec_dict.items()))
        else:
            test_num = self.exp_args.test_num
        for user, items in tqdm.tqdm(list(self.rec_dict.items())[:test_num]):
            items = self.rec_dict[user]
            margin_item = items[self.exp_args.rec_k]
            margin_score = self.base_model(self.user_feature_matrix[user].unsqueeze(0), 
                            self.item_feature_matrix[margin_item].unsqueeze(0)).squeeze()
            if self.exp_args.user_mask:
                # mask_vec = self.generate_mask(user)
                mask_vec = torch.where(self.user_feature_matrix[user]>0, 1., 0.).unsqueeze(0)  # only choose exps from the user cared aspects
            else:
                mask_vec = torch.ones(self.rec_dataset.feature_num, device=self.device).unsqueeze(0)
            for item in items[: self.exp_args.rec_k]:
                explanation_features, exp_num, exp_complexity = self.explain(
                    self.user_feature_matrix[user], 
                    self.item_feature_matrix[item], 
                    margin_score,
                    mask_vec)
                if explanation_features is None:
                    # print('no explanation for user %d and item %d' % (user, item))
                    no_exp_count += 1
                else:
                    self.u_i_exp_dict[(user, item)] = explanation_features
                    exp_nums.append(exp_num)
                    exp_complexities.append(exp_complexity)
        print('ave num: ', np.mean(exp_nums), 'ave complexity: ', np.mean(exp_complexities))
        return True
    
    def explain(self, user_feature, item_feature, margin_score, mask_vec):
        exp_generator = EXPGenerator(
            self.rec_dataset, 
            self.base_model, 
            user_feature, 
            item_feature, 
            margin_score, 
            mask_vec,
            self.device, 
            self.exp_args).to(self.device)

        # optimization
        optimizer = torch.optim.SGD(exp_generator.parameters(), lr=self.exp_args.lr, weight_decay=0)
        exp_generator.train()
        lowest_loss = None
        lowest_bpr = None
        lowest_l2 = 0
        optimize_delta = None
        score = exp_generator()
        bpr, l2, l1, loss = exp_generator.loss(score)
        # print('init: ', 0, '  train loss: ', loss, '  bpr: ', bpr, '  l2: ', l2, '  l1: ', l1)
        lowest_loss = loss
        optimize_delta = exp_generator.delta.detach().to('cpu').numpy()
        lowest_l2 = l2
        for epoch in range(self.exp_args.step):
            exp_generator.zero_grad()
            score = exp_generator()
            bpr, l2, l1, loss = exp_generator.loss(score)
            # if epoch % 100 == 0:
            #     print(
            #         'epoch', epoch,
            #         'bpr: ', bpr,
            #         'l2: ', l2,
            #         'l1', l1,
            #         'loss', loss)

            loss.backward()
            optimizer.step()
            if loss < lowest_loss:
                lowest_loss = loss
                lowest_l2 = l2
                lowest_bpr = bpr
                optimize_delta = exp_generator.delta.detach().to('cpu').numpy()
        if lowest_bpr >= self.exp_args.lam * self.exp_args.alp:
            explanation_features = None 
            exp_num = None
            exp_complexity = None
        else:
            # optimize_delta = exp_generator.delta.detach().to('cpu').numpy()
            explanation_features = np.argwhere(optimize_delta < - self.exp_args.mask_thresh).squeeze(axis=1)
            if len(explanation_features) == 0:
                explanation_features = np.array([np.argmin(optimize_delta)])
            exp_num = len(explanation_features)
            exp_complexity = lowest_l2.to('cpu').detach().numpy() + self.exp_args.gam * exp_num
        return explanation_features, exp_num, exp_complexity
    
    def user_side_evaluation(self):
        from utils.evaluate_functions import evaluate_user_perspective, evaluate_model_perspective
        ave_pre, ave_rec, ave_f1 = evaluate_user_perspective(self.user_perspective_test_data, self.u_i_exp_dict)
        print('user\'s perspective:')
        print('ave pre: ', ave_pre, '  ave rec: ', ave_rec, '  ave f1: ', ave_f1)
    
    def model_side_evaluation(self):
        from utils.evaluate_functions import evaluate_model_perspective
        ave_pn, ave_ps, ave_fns = evaluate_model_perspective(
            self.rec_dict,
            self.u_i_exp_dict,
            self.base_model,
            self.rec_dataset.user_feature_matrix,
            self.rec_dataset.item_feature_matrix,
            self.exp_args.rec_k,
            self.device)
        print('model\'s perspective:')
        print('ave PN: ', ave_pn, '  ave PS: ', ave_ps, '  ave F_{NS}: ', ave_fns)  


class EXPGenerator(torch.nn.Module):
    def __init__(self, rec_dataset, base_model, user_feature, item_feature, margin_score, mask_vec, device, exp_args):
        super(EXPGenerator, self).__init__()
        self.rec_dataset = rec_dataset
        self.base_model = base_model
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.margin_score = margin_score
        self.mask_vec = mask_vec
        self.device = device
        self.exp_args = exp_args
        self.feature_range = [0, 5]  # hard coded, should be improved later
        self.delta_range = self.feature_range[1] - self.feature_range[0]  # the maximum feature value.
        self.delta = torch.nn.Parameter(
            torch.FloatTensor(len(self.user_feature)).uniform_(-self.delta_range, 0))

    def get_masked_item_feature(self):
        item_feature_star = torch.clamp(
            (self.item_feature + torch.clamp((self.delta * self.mask_vec), -self.delta_range, 0)),
            self.feature_range[0], self.feature_range[1])
        return item_feature_star
    
    def forward(self):
        item_feature_star = self.get_masked_item_feature()
        score = self.base_model(self.user_feature.unsqueeze(0), item_feature_star)
        return score
    
    def loss(self, score):
        bpr = torch.nn.functional.relu(self.exp_args.alp + score - self.margin_score) * self.exp_args.lam
        l2 = torch.linalg.norm(self.delta)
        l1 = torch.linalg.norm(self.delta, ord=1) * self.exp_args.gam
        loss = l2 + bpr + l1
        return bpr, l2, l1, loss