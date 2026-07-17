# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a supervised machine learning classification model trained to predict whether a person’s income is greater than 50K per year based on census data. The model uses a `RandomForestClassifier` from scikit-learn. Categorical features are encoded using `OneHotEncoder`, and the target label is processed using `LabelBinarizer`.

The model was developed as part of the Udacity Machine Learning DevOps Engineer Nanodegree project on deploying a machine learning model with FastAPI.

## Intended Use

The intended use of this model is educational. It is meant to demonstrate a complete machine learning workflow including data preparation, model training, evaluation, slice-based analysis, testing, and API deployment.

This model should not be used for real-world decision-making in areas such as hiring, compensation, lending, insurance, legal decisions, or any other high-impact domain affecting individuals.

## Training Data

The training data comes from the census dataset provided in the project starter code (`census.csv`). The dataset contains demographic and employment-related features, including:

- age
- workclass
- fnlgt
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

The target variable is salary, indicating whether income is `>50K` or `<=50K`.

Before training, the data was cleaned to remove extra spaces from column names and categorical values. The dataset was then split into training and test sets.

## Evaluation Data

The evaluation data is a held-out test split from the same census dataset. This test set was used to evaluate the trained model after fitting on the training data.

In addition to overall evaluation, slice-based performance was computed for the `education` feature to better understand model behavior across subgroups.

## Metrics

The model was evaluated using the following metrics:

- Precision
- Recall
- F1 score

These metrics were computed using the `compute_model_metrics()` function.

Overall model performance on the full test set should be reported here if available:

- Overall model performance: Precision = [add overall value], Recall = [add overall value], F1 = [add overall value]

Slice-based performance for the `education` feature included the following results:

- Some-college: Precision = 0.6857, Recall = 0.5199, F1 = 0.5914
- HS-grad: Precision = 0.6594, Recall = 0.4377, F1 = 0.5261
- Bachelors: Precision = 0.7523, Recall = 0.7289, F1 = 0.7404
- Masters: Precision = 0.8271, Recall = 0.8551, F1 = 0.8409
- Assoc-acdm: Precision = 0.7000, Recall = 0.5957, F1 = 0.6437
- 7th-8th: Precision = 0.0000, Recall = 0.0000, F1 = 0.0000
- 11th: Precision = 1.0000, Recall = 0.2727, F1 = 0.4286
- Assoc-voc: Precision = 0.6471, Recall = 0.5238, F1 = 0.5789
- Prof-school: Precision = 0.8182, Recall = 0.9643, F1 = 0.8852
- 9th: Precision = 1.0000, Recall = 0.3333, F1 = 0.5000
- 5th-6th: Precision = 1.0000, Recall = 0.5000, F1 = 0.6667
- 10th: Precision = 0.4000, Recall = 0.1667, F1 = 0.2353
- Doctorate: Precision = 0.8644, Recall = 0.8947, F1 = 0.8793
- 12th: Precision = 1.0000, Recall = 0.4000, F1 = 0.5714
- 1st-4th: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000
- Preschool: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000

The full slice-based output is saved in `starter/slice_output.txt`.

## Ethical Considerations

This model is trained on census income data that includes sensitive demographic and social features such as race, sex, marital status, and native country. Because the model learns from historical data, it may reflect or amplify biases present in that data.

As a result, predictions may differ across groups in ways that are unfair or discriminatory. This makes the model inappropriate for real-world use in decisions that affect people’s opportunities, rights, or access to resources.

Any practical use of a similar model would require fairness analysis, bias mitigation, transparency, and human oversight.

## Caveats and Recommendations

This model has several limitations:

- It was built for instructional purposes rather than production use.
- It is trained on a specific census dataset that may be outdated or unrepresentative.
- Some slice groups have very small sample sizes, which can make their metrics unstable or misleading.
- The model has not been extensively tuned or validated beyond the project requirements.

Recommendations:

- Do not use this model for real-world decision-making.
- Perform fairness and bias analysis across multiple demographic slices before any practical deployment.
- Use larger and more representative datasets for production-oriented systems.
- Add stronger monitoring, validation, and documentation if extending this workflow further.
