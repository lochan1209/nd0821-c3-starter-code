# Put the code for your API here.
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import load_model, load_object, inference


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = load_model(MODEL_DIR / "model.pkl")
encoder = load_object(MODEL_DIR / "encoder.pkl")
lb = load_object(MODEL_DIR / "lb.pkl")
'''
model = load_model("starter/model/model.pkl")
encoder = load_object("starter/model/encoder.pkl")
lb = load_object("starter/model/lb.pkl")
'''
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


class CensusData(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(data: CensusData):
    input_data = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }])

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)[0]
    prediction = ">50K" if pred == 1 else "<=50K"

    return {"prediction": prediction}
