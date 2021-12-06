import torch
import pickle
import os
from pathlib import Path
from utils.argument_yelp import arg_parse_exp_optimize
from models.models import BaseRecModel, ExpOptimizationModel


def generate_explanation(exp_args):
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    # import dataset
    with open(os.path.join(exp_args.data_obj_path, exp_args.dataset + "_dataset_obj.pickle"), 'rb') as inp:
        rec_dataset = pickle.load(inp)
    
    base_model = BaseRecModel(rec_dataset.feature_num).to(device)
    base_model.load_state_dict(torch.load(os.path.join(exp_args.base_model_path, exp_args.dataset+"_logs", "model.model")))
    base_model.eval()
    #  fix the rec model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create optimization model
    opt_model = ExpOptimizationModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        device = device,
        exp_args=exp_args,
    )

    opt_model.generate_explanation()
    opt_model.user_side_evaluation()
    opt_model.model_side_evaluation()
    Path(exp_args.save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_args.save_path, exp_args.dataset + "_explanation_obj.pickle"), 'wb') as outp:
        pickle.dump(opt_model, outp, pickle.HIGHEST_PROTOCOL)
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    exp_args = arg_parse_exp_optimize()
    generate_explanation(exp_args)