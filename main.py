from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("logistic_pipeline.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    data_to_predict: list[list] = [[63,1,3,145,233,1,0,150,0,2.3,0,0,1,1]]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        list_data = request.data_to_predict
        df_data = DataFrame(list_data, columns=['age','trtbps','chol','thalachh','oldpeak','caa','sex','cp','fbs','restecg','slp','exng','thall','output'])

        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sum")
def sum(param1: float, param2: float):
    try:
        result = param1 + param2
        return {"param1": param1, "param2": param2, "sum": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {'Universidad EIA': 'MLOps'}
