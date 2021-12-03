

def dataset_init(preprocessing_args):
	if preprocessing_args.dataset == "yelp":
		from scripts.preprocessing.yelp_dataset_preprocessing import yelp_preprocessing
		rec_dataset = yelp_preprocessing(preprocessing_args)
	elif preprocessing_args.dataset == "cell_phones" or "kindle_store" or "electronic" or "cds_and_vinyl":
		from scripts.preprocessing.amazon_dataset_preprocessing import amazon_preprocessing
		rec_dataset = amazon_preprocessing(preprocessing_args)
	return rec_dataset
