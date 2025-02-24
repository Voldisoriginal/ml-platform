# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, ValidationError
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import io
import uuid
import os
import joblib
import json
# ML
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Database
import databases
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
# Celery
from celery import Celery, states
from celery.exceptions import Ignore
import time
#from .main_inference import app as inference_app # для интеграции
#import docker  # УДАЛЯЕМ отсюда
import threading
import datetime

# Импортируем из нашего нового модуля
from docker_utils import (get_running_containers,
                           start_inference_container,
                           stop_inference_container, MAX_CONTAINERS)
import docker # для celery
# --- Database Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL",
                                "postgresql://user:password@db:5432/dbname")  # Из переменных окружения (Docker Compose)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()



# --- Database Models ---
class DBModel(Base):  # Таблица для хранения моделей
    __tablename__ = "trained_models"

    id = Column(String, primary_key=True, index=True)
    model_type = Column(String)
    params = Column(JSON)
    metrics = Column(JSON)
    train_settings = Column(JSON)
    target_column = Column(String)
    dataset_filename = Column(String)  # filename


# --- Celery Configuration ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", 'redis://redis:6379/0')

celery = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_redirect_stdouts=True,
    worker_redirect_stdouts_level='DEBUG'
)


# --- Pydantic Models ---
class TrainSettings(BaseModel):
    train_size: float = 0.7
    random_state: int = 42

    @validator('train_size')
    def train_size_between_0_and_1(cls, v):
        if not 0 < v < 1:
            raise ValueError('train_size must be between 0 and 1')
        return v


class ModelParams(BaseModel):
    model_type: str
    params: Dict[str, Any] = {}

    @validator("params")
    def validate_params(cls, value, values):
        model_type = values.get("model_type")
        if model_type == "LinearRegression":
            pass
        elif model_type == "DecisionTreeRegressor":
            if "max_depth" in value:
                if not isinstance(value["max_depth"], int) or value["max_depth"] <= 0:  # type: ignore
                    raise ValueError("max_depth must be a positive integer")
            if 'min_samples_split' in value:
                if (not isinstance(value['min_samples_split'], int) or value['min_samples_split'] <= 0) and (
                        not isinstance(value['min_samples_split'], float) or not 0 < value[
                    'min_samples_split'] <= 1):  # type: ignore
                    raise ValueError("min_samples_split must be a positive integer or a float between 0 and 1")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return value


class TrainedModel(BaseModel):
    id: str
    model_type: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    train_settings: TrainSettings
    target_column: str
    dataset_filename: str


# --- FastAPI App ---
app = FastAPI()

# --- CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- File Storage ---
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# --- Database Helpers ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---Celery Tasks ---
import traceback
from celery import states
from celery.exceptions import Ignore
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@celery.task(bind=True, name='remove_container_task')
def remove_container_task(self, container_id):
    """Задача Celery для остановки и удаления контейнера."""
    try:
        stop_inference_container(container_id) # Используем функцию
        logger.info(f"Container removed: {container_id}")
    except Exception as e:
        logger.error(f"Error removing container {container_id}: {e}")
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),

            }
        )
        raise Ignore()

@celery.task(bind=True, name='train_model_task')
def train_model_task(self, dataset_filename: str, target_column: str,
                     train_settings: Dict, model_params: Dict):

    try:
        # 1. Загрузка датасета
        logger.debug(f"Загрузка датасета: {dataset_filename}")
        filepath = os.path.join(UPLOAD_FOLDER, dataset_filename)
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            exc_info = traceback.format_exc()

            self.update_state(
                state=states.FAILURE,
                meta={
                    'exc_type': type(e).__name__,
                    'exc_message': str(e),
                    'exc_traceback': exc_info,
                }
            )
            raise Ignore()
        logger.debug(f"Проверка целевой колонки: {target_column}")
        if target_column not in df.columns:
            logger.error(f"Целевая колонка отсутствует в датасете: {target_column}")
            exc_info = traceback.format_exc()
            self.update_state(
                state=states.FAILURE,
                meta={
                    'exc_type': 'KeyError',
                    'exc_message': "Target column not in df",
                    'exc_traceback': exc_info
                }
            )

            raise Ignore()

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # 2. Преобразование настроек обучения
        logger.debug(f"Преобразование настроек обучения: {train_settings}")
        train_settings_obj = TrainSettings(**train_settings)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_settings_obj.train_size,
            random_state=train_settings_obj.random_state
        )

        # 3. Обучение модели
        logger.debug(f"Преобразование параметров модели: {model_params}")
        model_params_obj = ModelParams(**model_params)
        if model_params_obj.model_type == "LinearRegression":
            model = LinearRegression(**model_params_obj.params)
        elif model_params_obj.model_type == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(**model_params_obj.params)
        else:
            logger.error(f"Неподдерживаемый тип модели: {model_params_obj.model_type}")
            exc_info = traceback.format_exc()
            self.update_state(
                state=states.FAILURE,
                meta={
                    'exc_type': 'ValueError',
                    'exc_message': "Unsupported model type",
                    'exc_traceback': exc_info
                }
)
            raise Ignore()

        logger.debug("Обучение модели...")
        model.fit(X_train, y_train)
        logger.debug("Обучение завершено.")

        # 4. Оценка модели
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred)
        }
        logger.debug(f"Метрики: {metrics}")

        # 5. Сохранение модели
        model_id = str(uuid.uuid4())
        model_filename = f"{model_id}.joblib"
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        logger.debug(f"Сохранение модели: {model_path}")
        joblib.dump(model, model_path)

        # Сохранение в БД
        logger.debug("Сохранение в БД...")
        with SessionLocal() as db:
            db_model = DBModel(
                id=model_id,
                model_type=model_params_obj.model_type,
                params=model_params_obj.params,
                metrics=metrics,
                train_settings=train_settings,
                target_column=target_column,
                dataset_filename=dataset_filename,
            )
            db.add(db_model)
            db.commit()
            result = {
                'id': db_model.id,
                'model_type': db_model.model_type,
                'params': db_model.params,
                'metrics': db_model.metrics,
                'train_settings': db_model.train_settings,
                'target_column': db_model.target_column,
                'dataset_filename': db_model.dataset_filename,
            }
        logger.debug("Сохранение в БД завершено.")
        return result

    except Exception as e:
        logger.exception("Произошла ошибка в задаче Celery:")
        exc_info = traceback.format_exc()

        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'exc_traceback': exc_info,
            }
        )
        raise Ignore()


    # --- API Endpoints ---

@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")  # Log filename
    print(f"File content type {file.content_type}")
    try:
        if file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="Invalid file type.  Must be CSV.")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        filename = f"{uuid.uuid4()}.csv"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(contents)
        return {"filename": filename, "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/")
async def train_model(
    dataset_filename: str = Form(...),
    target_column: str = Form(...),
    train_settings: str = Form(...),
    model_params: str = Form(...),
):
    try:
        train_settings_dict = json.loads(train_settings)
        model_params_dict = json.loads(model_params)
        if not isinstance(train_settings_dict, dict):
            raise TypeError("train_settings must be a dictionary")
        if not isinstance(model_params_dict, dict):
            raise TypeError("model_params must be a dictionary")
        TrainSettings(**train_settings_dict)
        ModelParams(**model_params_dict)

    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = train_model_task.delay(dataset_filename, target_column, train_settings_dict, model_params_dict)
    return {"task_id": task.id}


@app.get("/train_status/{task_id}")
async def get_train_status(task_id: str):
    task_result = celery.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result,
    }
    if task_result.status == states.FAILURE:
        if task_result.result and isinstance(task_result.result, dict):
            result['error'] = {
                'exc_type': task_result.result.get('exc_type'),
                'exc_message': task_result.result.get('exc_message'),
                'exc_traceback': task_result.result.get('exc_traceback', "No traceback"),
            }
        else:
            result['error'] = {
                'exc_type':  "UnknownError",
                'exc_message': str(task_result.result) if task_result.result else "Unknown error",
                'exc_traceback': "No traceback available",

            }
    return result


@app.get("/trained_models/", response_model=List[TrainedModel])
async def get_trained_models(db: Session = Depends(get_db)):
    db_models = db.query(DBModel).all()
    return [TrainedModel(**db_model.__dict__) for db_model in db_models]


@app.post("/predict/{model_id}")
async def predict(model_id: str, data: List[Dict[str, Union[float, int, str]]], db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.joblib")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        input_df = pd.DataFrame(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")

    try:
        predictions = model.predict(input_df).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {"predictions": predictions}


@app.get("/features/{model_id}", response_model=List[str])
async def get_model_features(model_id: str, db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_path = os.path.join(UPLOAD_FOLDER, db_model.dataset_filename)
    try:
        df = pd.read_csv(dataset_path)
        feature_names = df.drop(columns=[db_model.target_column]).columns.tolist()
        return feature_names
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")


@app.get("/download_dataset/{filename}")
async def download_dataset(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    with open(filepath, "rb") as f:
        content = f.read()

    return Response(content=content, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })


# --- Inference Container Management ---

@app.post("/start_inference/{model_id}")
async def start_inference(model_id: str, db: Session = Depends(get_db)):
    """Запускает контейнер для инференса модели."""
    running_containers = get_running_containers()

    if model_id in running_containers:
        return {
            "message": "Inference container is already running for this model.",
            "container_url": f"/inference/{model_id}",
        }

    if len(running_containers) >= MAX_CONTAINERS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many inference containers running.  Max: {MAX_CONTAINERS}",
        )

    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = os.path.join(MODEL_FOLDER, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        container_id, host_port = start_inference_container(model_id, model_path)

        if container_id is None or host_port is None:
            raise HTTPException(status_code=500, detail="Failed to start inference container.")

        # Запускаем задачу Celery для удаления контейнера через 1 час (3600 секунд)
        remove_container_task.apply_async((container_id,), countdown=3600)

        return {
            "message": "Inference container started",
            "container_url": f"/inference/{model_id}",
            "container_port": host_port, # type: ignore
        }
    except Exception as e:
        logger.exception(f"Failed to start inference container for model {model_id}")  # Логируем
        raise HTTPException(status_code=500, detail=f"Failed to start inference container. Error: {str(e)}")


@app.get("/inference/{model_id}")
async def inference_endpoint(model_id: str, db: Session = Depends(get_db)):
    """
    Предоставляет информацию о контейнере и URL для инференса.
    """
    running_containers = get_running_containers()
    if model_id not in running_containers:
        raise HTTPException(status_code=404, detail="Inference container not found or expired.")

    container_id = running_containers[model_id]
    docker_client = docker.from_env()

    try:
      container = docker_client.containers.get(container_id)
      container.reload()  # Обновляем информацию о контейнере
      ports = container.ports
      host_port = int(ports['8000/tcp'][0]['HostPort']) # type: ignore

      return {
        "message": "Inference container is running",
        "upload_url": f"http://localhost:{host_port}/predict_uploadfile/",
        "json_url": f"http://localhost:{host_port}/predict/"
      }
    except docker.errors.NotFound as e:
       logger.warning(f"Container not found.")
       raise HTTPException(status_code=404, detail=f"Container not found{str(e)}")
    except Exception as ex:
       logger.warning(f"Container error.")
       raise HTTPException(status_code=500, detail=f"Container error: {str(ex)}")

@app.delete("/stop_inference/{model_id}")
async def stop_inference(model_id: str):
    """Останавливает контейнер инференса."""
    running_containers = get_running_containers()

    if model_id not in running_containers:
        raise HTTPException(status_code=404, detail="Inference container not found.")

    container_id = running_containers[model_id]
    try:
        stop_inference_container(container_id)  # Используем функцию
        return {"message": "Inference container stopped."}
    except Exception as e:
        logger.exception(f"Failed to stop inference container for model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to stop inference container. Error: {str(e)}")

Base.metadata.create_all(bind=engine)

