from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import yaml
from src.logger import logger
from src.infer import Predictor

app = FastAPI(
    title="Predictor API",
    description="API to serve predictions from trained models.",
    version="1.0"
)

MODEL_CONFIG_PATH = "configs/model_confs.yaml"
DEVICE = None
try:
    with open(MODEL_CONFIG_PATH, "r") as file:
        MODEL_CONFS = yaml.safe_load(file)
    logger.info("Loaded model configurations successfully.")
except Exception as e:
    logger.error(f"Failed to load model configurations: {str(e)}")
    raise RuntimeError("Error loading model configurations. Check configs/model_confs.yaml.")

predictors = {}


def get_predictor(model_type: str = "RegressionNN"):
    """
    Lazily instantiate the predictor based on model type.
    """
    if model_type not in MODEL_CONFS:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    if model_type not in predictors:
        try:
            config_path = MODEL_CONFS[model_type]["config"]
            checkpoint_path = MODEL_CONFS[model_type]["checkpoint"]
            predictors[model_type] = Predictor(config_path, checkpoint_path, device=DEVICE)
            logger.info(f"Loaded {model_type} predictor successfully.")
        except Exception as e:
            logger.error(f"Failed to load {model_type} predictor: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

    return predictors[model_type]


class PredictionRequest(BaseModel):
    features: dict
    model_type: str = "RegressionNN"  # Default model is RegressionNN


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        predictor = get_predictor(request.model_type)
        prediction = predictor.infer(request.features)
        logger.info(f"Prediction successful for input: {request.features} using {request.model_type}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


class BatchPredictionRequest(BaseModel):
    data: dict  # Each key is a column name, and each value is a list of feature values.
    model_type: str = "RegressionNN"


@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    try:
        import pandas as pd
        predictor = get_predictor(request.model_type)
        df = pd.DataFrame(request.data)
        predictions = predictor.batch_infer(df)
        logger.info(f"Batch prediction successful using {request.model_type}")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device for model execution")
    args = parser.parse_args()
    DEVICE = args.device
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )
