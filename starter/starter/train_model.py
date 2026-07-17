import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    save_object,
)


def compute_slice_metrics(data, feature, model, encoder, lb, categorical_features, output_path):
    with open(output_path, "w") as f:
        for value in data[feature].unique():
            slice_df = data[data[feature] == value]

            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            line = (
                f"{feature}={value} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"fbeta={fbeta:.4f}"
            )

            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    data_path = "starter/data/census.csv"
    model_dir = "starter/model"
    slice_output_path = "starter/slice_output.txt"

    os.makedirs(model_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Overall precision: {precision:.4f}")
    print(f"Overall recall: {recall:.4f}")
    print(f"Overall fbeta: {fbeta:.4f}")

    save_model(model, f"{model_dir}/model.pkl")
    save_object(encoder, f"{model_dir}/encoder.pkl")
    save_object(lb, f"{model_dir}/lb.pkl")

    compute_slice_metrics(
        test,
        feature="education",
        model=model,
        encoder=encoder,
        lb=lb,
        categorical_features=cat_features,
        output_path=slice_output_path,
    )
