import traceback
import logging
import json
import time
import uuid
import os
import io
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, validator, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Response
import joblib
# ML
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Database
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, LargeBinary, DateTime
from datetime import datetime
# Celery
from celery import Celery, states
from celery.exceptions import Ignore
# Multiprocessing
import multiprocessing
from multiprocessing import Process, Queue, Event

# --- Database Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@db:5432/dbname")
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
engine = create_engine(DATABASE_URL, pool_pre_ping=True)  # Важно для multiprocessing
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
    model_data = Column(LargeBinary)
    start_time = Column(DateTime, default=datetime.utcnow)  # Добавлено: время начала обучения


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, unique=True, index=True)  # Add unique constraint
    upload_date = Column(DateTime, default=datetime.utcnow)
    columns = Column(JSON)  # Store column names as JSON

    def __repr__(self):
        return f"<Dataset(id={self.id}, filename={self.filename}, upload_date={self.upload_date})>"

# !!! Создание таблиц !!!
Base.metadata.create_all(bind=engine)

# --- Celery Configuration ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", 'redis://redis:6379/0')

celery = Celery(__name__, broker=CELERY_BROKER_URL,
                backend=CELERY_RESULT_BACKEND)
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
                    raise ValueError(
                        "min_samples_split must be a positive integer or a float between 0 and 1")
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
    start_time: datetime # Добавлено

class DatasetModel(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    columns: List[str]

    class Config:
        orm_mode = True

class RunningModel(BaseModel): # Добавлено для списка запущенных моделей
    model_id: str
    dataset_filename: str
    target_column: str
    model_type: str
    metrics: Dict[str, float]
    status: str
    api_url: str

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Database Helpers ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---Celery Tasks ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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
            logger.error(
                f"Неподдерживаемый тип модели: {model_params_obj.model_type}")
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
        model_id = str(uuid.uuid4())
        start_time = datetime.utcnow() #Фиксируем время начала

        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_data = buffer.getvalue()

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
                model_data=model_data,  # Сохраняем бинарные данные
                start_time = start_time
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
                'start_time': db_model.start_time
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


# --- Inference Process Management ---

class InferenceProcess:
    def __init__(self, model_id: str, model_data: bytes, result_queue: Queue):
        self.model_id = model_id
        self.model_data = model_data
        self.model: Optional[Any] = None
        self.process: Optional[Process] = None
        self.result_queue = result_queue
        self.input_queue = Queue()  # Queue for input data
        self.stop_event = Event()  # Use multiprocessing.Event

    def _run(self):
        logger.info(f"Inference process started for model {self.model_id}")
        try:
            buffer = io.BytesIO(self.model_data)
            self.model = joblib.load(buffer)
        except Exception as e:
            logger.exception(f"Error loading model in process {self.model_id}:")
            self.result_queue.put({  # Put error on result queue
                "model_id": self.model_id,
                "error": str(e),
                "predictions": None
            })
            return

        while not self.stop_event.is_set():
            try:
                # Wait for data on the input queue with a timeout
                data = self.input_queue.get(timeout=1)
                if data is None:
                    continue
                input_df = pd.DataFrame(data)
                predictions = self.model.predict(input_df).tolist()
                self.result_queue.put({
                    "model_id": self.model_id,
                    "error": None,
                    "predictions": predictions
                })
            except multiprocessing.queues.Empty:  # Correct exception for Queue.get()
                if self.stop_event.is_set():
                    break  # Exit loop if stop event set during timeout
            except Exception as e:
                logger.exception(f"Error during prediction in process {self.model_id}:")
                self.result_queue.put({
                    "model_id": self.model_id,
                    "error": str(e),
                    "predictions": None
                })
                break  # Important: Exit on error, don't keep the process running

        logger.info(f"Inference process stopped for model {self.model_id}")

    def predict(self, data: List[Dict]):
        # Put the data on the input queue, do *not* access self.model here
        self.input_queue.put(data)

    def start(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self._run, daemon=True)
            self.process.start()
        else:
            raise Exception("Inference process already running")

    def stop(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()  # Set the stop event
            # Signal the process to stop by sending None
            self.input_queue.put(None)
            self.process.join()  # Wait for process to terminate
            self.process = None  # Clear process reference

        else:
            raise Exception("Inference process is not running.")


inference_processes: Dict[str, InferenceProcess] = {}


# --- API Endpoints ---


@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type {file.content_type}")
    try:
        if file.content_type != "text/csv":
            raise HTTPException(
                status_code=400, detail="Invalid file type.  Must be CSV.")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        # Generate unique filename using UUID + original filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Check for duplicate filenames
        existing_dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
        if existing_dataset:
            raise HTTPException(status_code=409, detail="File with this name already exists.")


        with open(filepath, "wb") as f:
            f.write(contents)

        # Store dataset info in the database
        new_dataset = Dataset(filename=filename, columns=df.columns.tolist())
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        return {"filename": filename, "columns": df.columns.tolist(), "id": new_dataset.id}
    except HTTPException as e:
      raise e
    except Exception as e:
        logger.exception("Error uploading dataset")
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

    task = train_model_task.delay(
        dataset_filename, target_column, train_settings_dict, model_params_dict)
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
                'exc_type': "UnknownError",
                'exc_message': str(task_result.result) if task_result.result else "Unknown error",
                'exc_traceback': "No traceback available",

            }
    return result


@app.get("/trained_models/", response_model=List[TrainedModel])
async def get_trained_models(db: Session = Depends(get_db)):
    db_models = db.query(DBModel).all()
    # Convert to Pydantic models, *excluding* model_data.
    return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_data'}) for db_model in db_models]

@app.get("/trained_models/search_sort", response_model=List[TrainedModel])
async def search_sort_trained_models(
    db: Session = Depends(get_db),
    search_query: Optional[str] = None,
    sort_by: Optional[str] = "start_time",  # Default sort by start_time
    sort_order: Optional[str] = "desc",  # Default sort order (asc/desc)
    model_type: Optional[str] = None,      # Filter by model type
    dataset_filename: Optional[str] = None # Filter by dataset filename
):
    query = db.query(DBModel)

    # Filtering
    if search_query:
        query = query.filter(
            (DBModel.id.ilike(f"%{search_query}%")) |
            (DBModel.dataset_filename.ilike(f"%{search_query}%")) |
            (DBModel.target_column.ilike(f"%{search_query}%"))
        )
    if model_type:
        query = query.filter(DBModel.model_type == model_type)
    if dataset_filename:
        query = query.filter(DBModel.dataset_filename.ilike(f"%{dataset_filename}%"))

    # Sorting
    if sort_by:
        if sort_by == "start_time":
            sort_column = DBModel.start_time
        elif sort_by =="r2_score" and sort_order:
             if sort_order == "asc":
                 query = query.order_by(sqlalchemy.asc(DBModel.metrics['r2_score'].cast(Float)))
             elif sort_order == "desc":
                 query = query.order_by(sqlalchemy.desc(DBModel.metrics['r2_score'].cast(Float)))
             return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_data'}) for db_model in query.all()]
        elif sort_by == "mse" and sort_order:
            if sort_order == "asc":
                query = query.order_by(sqlalchemy.asc(DBModel.metrics['mse'].cast(Float)))
            elif sort_order == "desc":
                query = query.order_by(sqlalchemy.desc(DBModel.metrics['mse'].cast(Float)))
            return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_data'}) for db_model in query.all()]

        else:  # Add more sorting options as needed
            sort_column = getattr(DBModel, sort_by, None)


        if sort_column is not None:
            if sort_order == "asc":
                query = query.order_by(sort_column.asc())
            elif sort_order == "desc":
                query = query.order_by(sort_column.desc())

    db_models = query.all()
    return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_data'}) for db_model in db_models]


@app.get("/features/{model_id}", response_model=List[str])
async def get_model_features(model_id: str, db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_path = os.path.join(UPLOAD_FOLDER, db_model.dataset_filename)
    try:
        df = pd.read_csv(dataset_path)
        feature_names = df.drop(
            columns=[db_model.target_column]).columns.tolist()
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

@app.get("/datasets/", response_model=List[DatasetModel])
async def list_datasets(db: Session = Depends(get_db)):
    """Lists all uploaded datasets."""
    datasets = db.query(Dataset).all()
    return datasets

@app.get("/dataset/{dataset_id}", response_model=DatasetModel)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieves details of a specific dataset by ID."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Deletes a dataset from the database and the filesystem."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete associated trained models first (cascading delete)
    db.query(DBModel).filter(DBModel.dataset_filename == dataset.filename).delete()


    # Delete the dataset file
    filepath = os.path.join(UPLOAD_FOLDER, dataset.filename)
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass  # File might have been deleted already
    except Exception as e:
        logger.exception(f"Error deleting file {filepath}")
        db.rollback()  # rollback incase some models where deleted.
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    # Delete the dataset record from the database
    db.delete(dataset)
    db.commit()
    return {"message": f"Dataset '{dataset.filename}' deleted"}



@app.post("/start_inference/{model_id}")
async def start_inference_endpoint(model_id: str, db: Session = Depends(get_db)):
    """Starts an inference process."""
    global inference_processes

    if model_id in inference_processes and inference_processes[model_id].process.is_alive():
        return {
            "message": "Inference process already running",
            "status": "running",
            "model_id": model_id,
            "upload_url": f"http://localhost:8000/predict/{model_id}",
            "json_url": "See docs on http://localhost:8000/docs#/default/predict_endpoint_predict__model_id__post"
        }

    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model or not db_model.model_data:
        raise HTTPException(
            status_code=404,
            detail="Model not found or model data is missing"
        )

    try:
        if len(db_model.model_data) == 0:
            raise ValueError("Empty model data")

        # Always create a *new* queue for each process
        result_queue = Queue()
        inference_process = InferenceProcess(
            model_id, db_model.model_data, result_queue
        )
        inference_process.start()
        inference_processes[model_id] = inference_process

    except Exception as e:
        logger.error(f"Process start failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start inference process: {str(e)}"
        )

    return {
        "message": "Inference process started",
        "status": "running",
        "model_id": model_id,
        "upload_url": f"http://localhost:8000/predict/{model_id}",
        "json_url": "See docs on http://localhost:8000/docs#/default/predict_endpoint_predict__model_id__post"
    }


@app.post("/predict/{model_id}")
async def predict_endpoint(model_id: str, data: List[Dict[str, Union[float, int, str]]], db: Session = Depends(get_db)):
    global inference_processes
    if model_id not in inference_processes or not inference_processes[model_id].process.is_alive():
        raise HTTPException(status_code=404, detail="Inference process not running or not found")

    result_queue = inference_processes[model_id].result_queue
    try:
        inference_processes[model_id].predict(data)  # Put data on input queue
        # Get result from the result queue, with a timeout
        result = result_queue.get(timeout=10)  # Wait up to 10 seconds

        if result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"predictions": result["predictions"]}

    except multiprocessing.queues.Empty:
        raise HTTPException(status_code=500, detail="Prediction timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/stop_inference/{model_id}")
async def stop_inference_endpoint(model_id: str):
    """Stops the inference process."""
    global inference_processes

    if model_id not in inference_processes:
        raise HTTPException(status_code=404, detail="Inference not found")

    if inference_processes[model_id].process is None or not inference_processes[model_id].process.is_alive():
        raise HTTPException(status_code=404, detail="Inference process is not running")
    try:
        inference_processes[model_id].stop()
        del inference_processes[model_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Inference process stopped."}


@app.get("/inference_status/{model_id}")
async def inference_status_endpoint(model_id: str):
    """Checks if the inference process is running."""
    global inference_processes
    if model_id not in inference_processes or not inference_processes[model_id].process.is_alive():
        return {"status": "not running"}
    else:
        return {"status": "running"}


@app.get("/running_models/", response_model=List[RunningModel]) # Новый эндпоинт
async def get_running_models():
    """Returns a list of currently running inference models."""
    running_models_list = []
    for model_id, inference_process in inference_processes.items():
        if inference_process.process and inference_process.process.is_alive():
            # Достаём информацию о модели из БД, т.к.  inference_process хранит только бинарник
            with SessionLocal() as db:
                db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
                if db_model:  # Защита от устаревших
                    running_models_list.append(
                        RunningModel(
                            model_id=model_id,
                            dataset_filename=db_model.dataset_filename,
                            target_column=db_model.target_column,
                            model_type=db_model.model_type,
                            metrics=db_model.metrics,
                            status="running",
                            api_url=f"http://localhost:8000/predict/{model_id}",
                        )
                    )
    return running_models_list
