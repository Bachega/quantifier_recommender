from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from utils import generate_train_test_data

if __name__ == "__main__":
    dataset_path = './datasets/'
    recommender = QuantifierRecommender(supervised=True)

    # recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    # recommender.save_meta_table('meta-features.csv')

    # quantifier_evaluator = QuantifierEvaluator()
    # QuantifierEvaluator.__quantifiers

    generate_train_test_data("./datasets")