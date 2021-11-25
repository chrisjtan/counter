import argparse

def arg_parser_pre_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="yelp")
    parser.add_argument("--sentires_dir", dest="sentires_dir", type=str, default="./datasets/Yelp/reviews.pickle", 
                        help="path to pre-extracted sentires data")
    parser.add_argument("--review_dir", dest="review_dir", type=str, default="./datasets/Yelp/yelp_academic_dataset_review.json", 
                        help="path to original review data")
    parser.add_argument("--user_thresh", dest="user_thresh", type=int, default=20, 
    help="remove users with reviews less than this threshold")
    parser.add_argument("--feature_thresh", dest="feature_thresh", type=int, default=2000, 
    help="remove the features mentioned less than this threshold")
    parser.add_argument("--training_ratio", dest="training_length", type=float, default=0.9, 
    help="the number of training items")
    parser.add_argument("--sample_ratio", dest="sample_ratio", type=int, default=2, 
                        help="the negative: positive sample ratio for training BPR loss")
    parser.add_argument("--neg_length", dest="neg_length", type=int, default=9, 
    help="num of sampled items in evaluation")
    parser.add_argument("--save_path", dest="save_path", type=str, default="./dataset_objs/", 
    help="The path to save the preprocessed dataset object")
    return parser.parse_args()


def arg_parse_train_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="yelp")
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='1e-5', help="L2 norm to the wights")
    parser.add_argument("--lr", dest="lr", type=float, default=0.0005, help="learning rate for training")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1000, help="training epoch")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=256, help="batch size for training base rec model")
    parser.add_argument("--embedding_length", dest="embedding_length", type=int, default=256, help="implicit feature length")
    return parser.parse_args()