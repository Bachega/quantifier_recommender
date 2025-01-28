import numpy as np

def PWK(test, clf):
    # Predict class labels for the given data
    predicted_labels = clf.predict(test)

    # Compute the distribution of predicted labels
    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)

    # Calculate the prevalence for each class
    class_prevalences = label_counts / label_counts.sum()

    # Map each class label to its prevalence
    prevalences = {label: prevalence for label, prevalence in zip(unique_labels, class_prevalences)}

    return prevalences[1]