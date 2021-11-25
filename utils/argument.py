import argparse

def arg_parser_pre_processing_yelp():
    parser = argparse.ArgumentParser()
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