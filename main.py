from quantifier_recommender import QuantifierRecommender


if __name__ == "__main__":
    dataset_path = './datasets/'
    recommender = QuantifierRecommender(supervised=True)

    # dt_list = load_datasets(dataset_path)
    # for dt in dt_list:
    #     y = dt.pop(dt.columns[-1])
    #     X = dt
    #     recommender.extract_and_append(X, y)
    # recommender.normalize_meta_table()

    recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    recommender.save_meta_table('meta-features.csv')