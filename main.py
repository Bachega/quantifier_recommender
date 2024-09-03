from quantifier_recommender import QuantifierRecommender
from quantifier_evaluator import QuantifierEvaluator
from utils import generate_train_test_data
from quantifier_evaluator import QuantifierEvaluator

if __name__ == "__main__":
    dataset_path = "./datasets"
    # recommender = QuantifierRecommender(supervised=True)

    # recommender.construct_meta_table(dataset_path=dataset_path, supervised=True)
    # recommender.save_meta_table('meta-features.csv')

    # quantifier_evaluator = QuantifierEvaluator()
    # QuantifierEvaluator.__quantifiers

    generate_train_test_data("./datasets")


    # qtf_eval = QuantifierEvaluator()
    # qtf_eval.evaluate("a", "b")
    # qtf_eval.save_processed_datasets_list()

    # processed_datasets_list = ["a.csv", "b.csv", "c.csv", "d.csv"]
    # with open("processed_datasets.json", 'w') as file:
    #     json.dump(processed_datasets_list, file, indent=4, ensure_ascii=False)


    