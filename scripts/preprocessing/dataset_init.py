

def dataset_init(preprocessing_args):
	if preprocessing_args.dataset == "yelp":
		from scripts.preprocessing.yelp_dataset_preprocessing import yelp_preprocessing
		rec_dataset = yelp_preprocessing(preprocessing_args)
	return rec_dataset
