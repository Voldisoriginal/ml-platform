import traceback
import logging
import json
import uuid
import os
import io
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, validator, Field, ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Response, Query
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
from sqlalchemy import create_engine, Column, String, Float, JSON, LargeBinary, DateTime
from datetime import datetime
# Celery
from celery import Celery, states
from celery.exceptions import Ignore
# Multiprocessing
import multiprocessing
from multiprocessing import Process, Queue, Event
import numpy as np # Понадобится для гистограмм и корреляции
from scipy import stats # Может понадобиться для стат. деталей боксплотов
from celery.result import AsyncResult # Импортируем для проверки статуса
# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Database Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@db:5432/dbname")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)  # Important for multiprocessing
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---
class DBModel(Base):  # Table for storing models
    __tablename__ = "trained_models"

    id = Column(String, primary_key=True, index=True)
    model_type = Column(String)
    params = Column(JSON)
    metrics = Column(JSON)
    train_settings = Column(JSON)
    target_column = Column(String)
    dataset_filename = Column(String)
    model_filename = Column(String)  # Store the model filename
    start_time = Column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, unique=True, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    columns = Column(JSON)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    author = Column(String, nullable=True)
    target_variable = Column(String, nullable=True)
    image_filename = Column(String, nullable=True)

    def __repr__(self):
        return f"<Dataset(id={self.id}, filename={self.filename}, upload_date={self.upload_date})>"


class DatasetVisualization(Base):
    __tablename__ = "dataset_visualizations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, sqlalchemy.ForeignKey("datasets.id"), index=True, nullable=False) # Связь с Dataset
    generated_at = Column(DateTime, default=datetime.utcnow)
    visualization_data = Column(JSON) # Здесь будем хранить JSON с данными для графиков
    status = Column(String, default="PENDING") # PENDING, SUCCESS, FAILURE
    error_message = Column(String, nullable=True) # Сообщение об ошибке

    dataset = sqlalchemy.orm.relationship("Dataset") # Связь для удобства

# Create tables
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
                if not isinstance(value["max_depth"], int) or value["max_depth"] <= 0:
                    raise ValueError("max_depth must be a positive integer")
            if 'min_samples_split' in value:
                if (not isinstance(value['min_samples_split'], int) or value['min_samples_split'] <= 0) and (
                        not isinstance(value['min_samples_split'], float) or not 0 < value[
                    'min_samples_split'] <= 1):
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
    start_time: datetime


class DatasetModel(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    columns: List[str] # Original list of column names
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    target_variable: Optional[str] = None
    imageUrl: Optional[str] = None
    # --- New fields for detail view ---
    file_size: Optional[int] = None # Size in bytes
    row_count: Optional[int] = None # Number of data rows (excluding header)
    column_types: Optional[Dict[str, str]] = None # Dictionary mapping column name to pandas dtype string

    class Config:
        orm_mode = True
        from_attributes = True  # Add this line!

class DatasetPreviewResponse(BaseModel):
    preview: List[Dict[str, Any]] # List of dictionaries representing rows
    column_types: Optional[Dict[str, str]] = None # Column types detected from preview

class RunningModel(BaseModel):
    model_id: str
    dataset_filename: str
    target_column: str
    model_type: str
    metrics: Dict[str, float]
    status: str
    api_url: str

class VisualizationDataResponse(BaseModel):
    id: str
    dataset_id: str
    generated_at: datetime
    visualization_data: Optional[Dict[str, Any]] # Данные для графиков
    status: str # PENDING, SUCCESS, FAILURE
    error_message: Optional[str] = None

    class Config:
        orm_mode = True # Для совместимости с SQLAlchemy
        from_attributes = True

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
        # 1. Load dataset
        logger.debug(f"Loading dataset: {dataset_filename}")
        dataset_path = os.path.join(UPLOADS_DIR, dataset_filename)
        try:
            with SessionLocal() as db:
                dataset = db.query(Dataset).filter(Dataset.filename == dataset_filename).first()

            if dataset:
                df = pd.read_csv(dataset_path)
            else:
                logger.error(f"Dataset loading error: File not found")
                exc_info = traceback.format_exc()
                self.update_state(
                    state=states.FAILURE,
                    meta={
                        'exc_type': "FileNotFoundError",
                        'exc_message': "File not found",
                        'exc_traceback': exc_info,
                    }
                )
                raise Ignore()

        except FileNotFoundError as e:
            logger.error(f"Dataset loading error: {e}")
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
        logger.debug(f"Checking target column: {target_column}")
        if target_column not in df.columns:
            logger.error(f"Target column missing in dataset: {target_column}")
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

        # 2. Transform train settings
        logger.debug(f"Transforming train settings: {train_settings}")
        train_settings_obj = TrainSettings(**train_settings)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_settings_obj.train_size,
            random_state=train_settings_obj.random_state
        )

        # 3. Train model
        logger.debug(f"Transforming model parameters: {model_params}")
        model_params_obj = ModelParams(**model_params)
        if model_params_obj.model_type == "LinearRegression":
            model = LinearRegression(**model_params_obj.params)
        elif model_params_obj.model_type == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(**model_params_obj.params)
        else:
            logger.error(
                f"Unsupported model type: {model_params_obj.model_type}")
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

        logger.debug("Training model...")
        model.fit(X_train, y_train)
        logger.debug("Training complete.")

        # 4. Evaluate model
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred)
        }
        logger.debug(f"Metrics: {metrics}")
        model_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Save the model to a file
        model_filename = f"{model_id}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)

        # Save to database
        logger.debug("Saving to database...")
        with SessionLocal() as db:
            db_model = DBModel(
                id=model_id,
                model_type=model_params_obj.model_type,
                params=model_params_obj.params,
                metrics=metrics,
                train_settings=train_settings,
                target_column=target_column,
                dataset_filename=dataset_filename,
                model_filename=model_filename,
                start_time=start_time
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
        logger.debug("Database save complete.")
        return result

    except Exception as e:
        logger.exception("An error occurred in the Celery task:")
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

@celery.task(bind=True, name='generate_visualizations_task')
def generate_visualizations_task(self, dataset_id: str):
    """
    Генерирует данные для визуализаций (гистограммы, боксплоты, корреляция)
    и сохраняет их в базу данных.
    """
    task_id = self.request.id # Получаем ID задачи Celery
    db: Session = SessionLocal()
    visualization_entry = None
    try:
        logger.info(f"[{task_id}] Starting visualization generation for dataset_id: {dataset_id}")

        # Найти датасет в БД
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise FileNotFoundError(f"Dataset with ID {dataset_id} not found in database.")

        # Создать или найти запись для визуализации
        visualization_entry = db.query(DatasetVisualization).filter(DatasetVisualization.dataset_id == dataset_id).first()
        if not visualization_entry:
            visualization_entry = DatasetVisualization(dataset_id=dataset_id, status="PENDING")
            db.add(visualization_entry)
        else:
            visualization_entry.status = "PENDING" # Сброс статуса при перезапуске
            visualization_entry.error_message = None
            visualization_entry.visualization_data = None # Очистить старые данные
        db.commit()
        db.refresh(visualization_entry)
        logger.info(f"[{task_id}] Visualization entry created/updated with ID: {visualization_entry.id}")


        # Загрузить данные
        dataset_path = os.path.join(UPLOADS_DIR, dataset.filename)
        if not os.path.exists(dataset_path):
             raise FileNotFoundError(f"Dataset file {dataset.filename} not found at {dataset_path}.")

        df = pd.read_csv(dataset_path)
        logger.info(f"[{task_id}] Loaded dataset {dataset.filename}, shape: {df.shape}")

        # Отбираем только числовые колонки
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numerical_cols:
             logger.warning(f"[{task_id}] No numerical columns found in dataset {dataset.filename}.")
             visualization_entry.status = "SUCCESS" # Успешно, но данных нет
             visualization_entry.visualization_data = {"histograms": [], "boxplots": [], "correlation_matrix": None}
             db.commit()
             return {"message": "No numerical columns found."}


        results = {
            "histograms": [],
            "boxplots": [],
            "correlation_matrix": None
        }

        # 1. Гистограммы
        logger.info(f"[{task_id}] Generating histograms for: {numerical_cols}")
        for col in numerical_cols:
            # Игнорируем колонки с одним уникальным значением или NaN
            if df[col].nunique(dropna=True) <= 1:
                logger.warning(f"[{task_id}] Skipping histogram for column '{col}' (<=1 unique value).")
                continue
            try:
                # Убираем NaN для построения гистограммы
                data = df[col].dropna()
                if data.empty:
                    logger.warning(f"[{task_id}] Skipping histogram for column '{col}' (all NaN).")
                    continue

                # Numpy histogram для получения данных
                counts, bin_edges = np.histogram(data, bins='auto') # 'auto' подбирает количество бинов

                results["histograms"].append({
                    "column": col,
                    "bins": bin_edges.tolist(), # Края бинов
                    "counts": counts.tolist()   # Количество в каждом бине
                })
            except Exception as e:
                 logger.error(f"[{task_id}] Error generating histogram for column '{col}': {e}", exc_info=True)
                 # Можно добавить информацию об ошибке в результат или пропустить колонку

        # 2. Боксплоты (Ящик с усами) - собираем статистику
        logger.info(f"[{task_id}] Generating boxplot stats for: {numerical_cols}")
        for col in numerical_cols:
             if df[col].nunique(dropna=True) < 1: # Пропускаем если только NaN
                 logger.warning(f"[{task_id}] Skipping boxplot for column '{col}' (all NaN or empty).")
                 continue
             try:
                # Используем describe для основных статистик
                stats_desc = df[col].describe()
                q1 = stats_desc.get('25%')
                median = stats_desc.get('50%')
                q3 = stats_desc.get('75%')
                min_val = stats_desc.get('min')
                max_val = stats_desc.get('max')

                # Выбросы обычно определяются как < Q1 - 1.5*IQR или > Q3 + 1.5*IQR
                # Для передачи в Chart.js обычно достаточно min, q1, median, q3, max
                # Chart.js сам может нарисовать или потребует плагин

                # Проверяем, что все значения не NaN/None перед добавлением
                if all(pd.notna(x) for x in [min_val, q1, median, q3, max_val]):
                    results["boxplots"].append({
                        "column": col,
                        "min": min_val,
                        "q1": q1,
                        "median": median,
                        "q3": q3,
                        "max": max_val
                        # Можно добавить 'mean': stats_desc.get('mean') если нужно
                    })
                else:
                     logger.warning(f"[{task_id}] Skipping boxplot for column '{col}' due to missing stats (likely all NaN).")

             except Exception as e:
                 logger.error(f"[{task_id}] Error generating boxplot stats for column '{col}': {e}", exc_info=True)


        # 3. Матрица корреляции (только если >1 числовой колонки)
        if len(numerical_cols) > 1:
            logger.info(f"[{task_id}] Generating correlation matrix.")
            try:
                correlation = df[numerical_cols].corr()
                # Заменяем NaN на None для JSON-совместимости
                correlation = correlation.where(pd.notnull(correlation), None)
                results["correlation_matrix"] = {
                    "columns": correlation.columns.tolist(),
                    "data": correlation.values.tolist()
                }
            except Exception as e:
                 logger.error(f"[{task_id}] Error generating correlation matrix: {e}", exc_info=True)
        else:
             logger.info(f"[{task_id}] Skipping correlation matrix (<=1 numerical column).")


        # Сохраняем результаты в БД
        visualization_entry.visualization_data = results
        visualization_entry.status = "SUCCESS"
        visualization_entry.generated_at = datetime.utcnow()
        db.commit()
        logger.info(f"[{task_id}] Successfully generated and saved visualizations for dataset {dataset_id}.")

        # Возвращаем ID записи визуализации (не сами данные, они в БД)
        return {"visualization_db_id": visualization_entry.id, "status": "SUCCESS"}

    except FileNotFoundError as e:
        logger.error(f"[{task_id}] File Not Found Error during visualization: {e}")
        if visualization_entry:
            visualization_entry.status = "FAILURE"
            visualization_entry.error_message = str(e)
            db.commit()
        # Не используем self.update_state, т.к. хотим вернуть ошибку через API статус
        raise Ignore() # Не повторять задачу Celery
    except Exception as e:
        logger.exception(f"[{task_id}] An unexpected error occurred during visualization:")
        if visualization_entry:
            visualization_entry.status = "FAILURE"
            visualization_entry.error_message = traceback.format_exc() # Сохраняем traceback
            db.commit()
        # Обновляем статус Celery для /train_status эндпоинта (если он будет использоваться для этого)
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'exc_traceback': traceback.format_exc(),
            }
        )
        raise Ignore() # Не повторять задачу Celery
    finally:
        db.close()


# --- Inference Process Management ---

class InferenceProcess:
    def __init__(self, model_id: str, model_filename: str, result_queue: Queue):
        self.model_id = model_id
        self.model_filename = model_filename
        self.model_path = os.path.join(MODELS_DIR, self.model_filename)
        self.model: Optional[Any] = None
        self.process: Optional[Process] = None
        self.result_queue = result_queue
        self.input_queue = Queue()
        self.stop_event = Event()

    def _run(self):
        logger.info(f"Inference process started for model {self.model_id}")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            logger.exception(f"Error loading model in process {self.model_id}:")
            self.result_queue.put({
                "model_id": self.model_id,
                "error": str(e),
                "predictions": None
            })
            return

        while not self.stop_event.is_set():
            try:
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
            except multiprocessing.queues.Empty:
                if self.stop_event.is_set():
                    break
            except Exception as e:
                logger.exception(f"Error during prediction in process {self.model_id}:")
                self.result_queue.put({
                    "model_id": self.model_id,
                    "error": str(e),
                    "predictions": None
                })
                #break

        logger.info(f"Inference process stopped for model {self.model_id}")

    def predict(self, data: List[Dict]):
        self.input_queue.put(data)

    def start(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self._run, daemon=True)
            self.process.start()
        else:
            raise Exception("Inference process already running")

    def stop(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.input_queue.put(None)
            self.process.join()
            self.process = None

        else:
            raise Exception("Inference process is not running.")


inference_processes: Dict[str, InferenceProcess] = {}


# --- API Endpoints ---


@app.post("/upload_dataset/")
async def upload_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        description: Optional[str] = Form(None),
        author: Optional[str] = Form(None),
        target_variable: Optional[str] = Form(None),
        image: Optional[UploadFile] = Form(None),
        db: Session = Depends(get_db)

):
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type {file.content_type}")

    try:
        if file.content_type != "text/csv":
            raise HTTPException(
                status_code=400, detail="Invalid file type.  Must be CSV.")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOADS_DIR, filename)

        existing_dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
        if existing_dataset:
            raise HTTPException(status_code=409, detail="File with this name already exists.")
        with open(filepath, "wb") as f:
            f.write(contents)

        image_filename = None
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image file type.")

            image_filename = f"{uuid.uuid4()}_{image.filename}"
            image_path = os.path.join(UPLOADS_DIR, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(await image.read())

        new_dataset = Dataset(filename=filename, columns=df.columns.tolist(),
                              name=name, description=description, author=author,
                              target_variable=target_variable, image_filename=image_filename)

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        return {"filename": filename, "columns": df.columns.tolist(), "id": new_dataset.id,
                'imageUrl': f"/dataset/{new_dataset.id}/image" if new_dataset.image_filename else None}


    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error uploading dataset")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/{dataset_id}", response_model=DatasetModel)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Gets detailed information about a specific dataset by its ID."""
    logger.debug(f"Fetching details for dataset ID: {dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logger.warning(f"Dataset with ID {dataset_id} not found in DB.")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Base info from DB
    image_url = f"/dataset/{dataset.id}/image" if dataset.image_filename else None
    file_size = None
    row_count = None
    column_types = None
    filepath = os.path.join(UPLOADS_DIR, dataset.filename)

    # Get file statistics if file exists
    if os.path.exists(filepath):
        try:
            file_size = os.path.getsize(filepath)
            logger.debug(f"File size for {dataset.filename}: {file_size} bytes")

            # Count rows efficiently (assuming header row exists)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for _ in f) - 1 # Subtract 1 for header
            logger.debug(f"Row count for {dataset.filename}: {row_count}")

            # Get column types using pandas on a sample
            df_sample = pd.read_csv(filepath, nrows=50) # Read a small sample
            column_types = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
            logger.debug(f"Column types detected for {dataset.filename}: {column_types}")

        except FileNotFoundError:
             logger.error(f"File not found on disk for dataset {dataset_id} at path {filepath}, though DB record exists.")
             # Don't raise 404 here, return DB info but mark stats as unavailable
             file_size = -1 # Indicate file missing
             row_count = -1
        except Exception as e:
            logger.error(f"Error getting file stats for {dataset.filename} (ID: {dataset_id}): {e}", exc_info=True)
            # Return DB info but indicate stats couldn't be retrieved
            file_size = -2 # Indicate error getting stats
            row_count = -2

    # Construct the response model using all gathered info
    response_data = DatasetModel(
        id=dataset.id,
        filename=dataset.filename,
        upload_date=dataset.upload_date,
        columns=dataset.columns, # List of column names from DB
        name=dataset.name,
        description=dataset.description,
        author=dataset.author,
        target_variable=dataset.target_variable,
        imageUrl=image_url,
        # Add the new stats fields
        file_size=file_size,
        row_count=row_count,
        column_types=column_types
    )
    logger.debug(f"Returning details for dataset ID: {dataset_id}")
    return response_data


@app.get("/dataset/{dataset_id}/image")
async def get_dataset_image(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset or not dataset.image_filename:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = os.path.join(UPLOADS_DIR, dataset.image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found on server")
    try:
        with open(image_path, "rb") as image_file:
            return Response(content=image_file.read(), media_type="image/jpeg")  # Or appropriate media type
    except Exception as e:
        logger.exception(f"Error reading image file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading image file: {e}")


# *** MODIFIED Endpoint to return list of datasets (using updated DatasetModel) ***
@app.get("/datasets/", response_model=List[DatasetModel])
async def list_datasets(db: Session = Depends(get_db)):
    """Lists all available datasets with basic information."""
    logger.debug("Fetching list of all datasets.")
    datasets_db = db.query(Dataset).order_by(Dataset.upload_date.desc()).all() # Order by upload date
    response_data = []
    for dataset in datasets_db:
        image_url = f"/dataset/{dataset.id}/image" if dataset.image_filename else None
        # *** Decision: DO NOT fetch file stats (size, rows, types) here ***
        # It would require reading every file and slow down the list view significantly.
        # Stats will be fetched on demand by the detail view endpoint (/dataset/{id}).
        dataset_model = DatasetModel(
             id=dataset.id,
             filename=dataset.filename,
             upload_date=dataset.upload_date,
             columns=dataset.columns,
             name=dataset.name,
             description=dataset.description,
             author=dataset.author,
             target_variable=dataset.target_variable,
             imageUrl=image_url,
             # File stats are intentionally left as None here
             file_size=None,
             row_count=None,
             column_types=None
         )
        response_data.append(dataset_model)

    logger.debug(f"Returning {len(response_data)} datasets.")
    return response_data

# *** NEW Endpoint for fetching dataset preview ***
@app.get("/dataset_preview/{filename}", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    filename: str, # Can be filename or dataset ID
    rows: int = Query(20, ge=1, le=100), # Get first N rows, limit query param
    db: Session = Depends(get_db) # Inject DB session to potentially look up by ID
    ):
    """
    Gets a preview (first N rows) of a dataset's content.
    Accepts either the dataset filename or the dataset ID.
    """
    logger.debug(f"Request for preview of '{filename}', rows={rows}")
    filepath = os.path.join(UPLOADS_DIR, filename)
    actual_filename = filename

    # Check if the file exists directly
    if not os.path.exists(filepath):
        logger.warning(f"Preview request: File '{filename}' not found directly. Checking if it's a dataset ID.")
        # If not found, check if 'filename' might be a dataset ID
        dataset_by_id = db.query(Dataset).filter(Dataset.id == filename).first()
        if dataset_by_id and dataset_by_id.filename:
            logger.debug(f"Found dataset by ID {filename}. Actual filename: {dataset_by_id.filename}")
            actual_filename = dataset_by_id.filename
            filepath = os.path.join(UPLOADS_DIR, actual_filename)
            if not os.path.exists(filepath):
                 logger.error(f"File path {filepath} for dataset ID {filename} not found on disk, inconsistent state.")
                 raise HTTPException(status_code=404, detail=f"Dataset file associated with ID {filename} not found on server.")
        else:
            logger.error(f"Preview request: Neither file nor dataset ID '{filename}' found.")
            raise HTTPException(status_code=404, detail="Dataset file or ID not found.")

    # Now we have a valid filepath
    try:
        logger.debug(f"Reading preview from: {filepath}")
        df_preview = pd.read_csv(filepath, nrows=rows)

        # Convert NaN/NaT to None for JSON compatibility
        df_cleaned = df_preview.where(pd.notnull(df_preview), None)
        preview_data = df_cleaned.to_dict(orient='records')

        # Get column types from the preview
        column_types = {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
        logger.debug(f"Preview generated successfully for {actual_filename}.")
        return DatasetPreviewResponse(preview=preview_data, column_types=column_types)

    except FileNotFoundError:
         # This should be caught by the initial checks, but handle defensively
         logger.error(f"FileNotFoundError during preview generation for {actual_filename} at {filepath}.")
         raise HTTPException(status_code=404, detail="Dataset file not found during preview generation.")
    except pd.errors.EmptyDataError:
         logger.warning(f"EmptyDataError during preview for {actual_filename}. Returning empty preview.")
         return DatasetPreviewResponse(preview=[], column_types={}) # Return empty list if file is empty/header only
    except Exception as e:
        logger.error(f"Error reading dataset preview for {actual_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading dataset preview: {e}")



@app.get("/placeholder.png")
async def get_placeholder_image():
    placeholder_path = os.path.join(BASE_DIR, "placeholder.png")
    if not os.path.exists(placeholder_path):
        # Create a very basic placeholder image on-the-fly if needed
        from PIL import Image, ImageDraw

        img = Image.new('RGB', (300, 200), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((100, 100), "Placeholder", fill=(50, 50, 50))
        img.save(placeholder_path)

    with open(placeholder_path, "rb") as image_file:
        return Response(content=image_file.read(), media_type="image/png")


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
    return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_filename'}) for db_model in
            db_models]



@app.get("/trained_models/search_sort", response_model=List[TrainedModel])
async def search_sort_trained_models(
        db: Session = Depends(get_db),
        search_query: Optional[str] = None,
        sort_by: Optional[str] = "start_time",
        sort_order: Optional[str] = "desc",
        model_type: Optional[str] = None,
        dataset_filename: Optional[str] = None,
        model_id: Optional[str] = None  # Добавляем новый параметр

):
    query = db.query(DBModel)

    if model_id:
        query = query.filter(DBModel.id == model_id)
    elif search_query:
        query = query.filter(
            (DBModel.id.ilike(f"%{search_query}%")) |
            (DBModel.dataset_filename.ilike(f"%{search_query}%")) |
            (DBModel.target_column.ilike(f"%{search_query}%"))
        )

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

    if sort_by:
        if sort_by == "start_time":
            sort_column = DBModel.start_time
        elif sort_by == "r2_score" and sort_order:
            if sort_order == "asc":
                query = query.order_by(sqlalchemy.asc(DBModel.metrics['r2_score'].cast(Float)))
            elif sort_order == "desc":
                query = query.order_by(sqlalchemy.desc(DBModel.metrics['r2_score'].cast(Float)))
            return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_filename'}) for db_model
                    in query.all()]
        elif sort_by == "mse" and sort_order:
            if sort_order == "asc":
                query = query.order_by(sqlalchemy.asc(DBModel.metrics['mse'].cast(Float)))
            elif sort_order == "desc":
                query = query.order_by(sqlalchemy.desc(DBModel.metrics['mse'].cast(Float)))
            return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_filename'}) for db_model
                    in query.all()]

        else:
            sort_column = getattr(DBModel, sort_by, None)

        if sort_column is not None:
            if sort_order == "asc":
                query = query.order_by(sort_column.asc())
            elif sort_order == "desc":
                query = query.order_by(sort_column.desc())

    db_models = query.all()
    return [TrainedModel(**{k: v for k, v in db_model.__dict__.items() if k != 'model_filename'}) for db_model in
            db_models]


@app.get("/features/{model_id}", response_model=List[str])
async def get_model_features(model_id: str, db: Session = Depends(get_db)):
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_path = os.path.join(UPLOADS_DIR, db_model.dataset_filename)
    try:
        df = pd.read_csv(dataset_path)
        feature_names = df.drop(
            columns=[db_model.target_column]).columns.tolist()
        return feature_names
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")


@app.get("/download_dataset/{filename}")
async def download_dataset(filename: str, db: Session = Depends(get_db)):
    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    dataset = db.query(Dataset).filter(Dataset.filename == filename).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="File not found in database")

    with open(filepath, "rb") as f:
        content = f.read()

    return Response(content=content, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })


@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    associated_models = db.query(DBModel).filter(DBModel.dataset_filename == dataset.filename).all()
    for model in associated_models:
        model_path = os.path.join(MODELS_DIR, model.model_filename)
        try:
            os.remove(model_path)
        except FileNotFoundError:
            pass
        db.delete(model)

    filepath = os.path.join(UPLOADS_DIR, dataset.filename)
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.exception(f"Error deleting file {dataset.filename}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    # Delete image
    if dataset.image_filename:
        image_path = os.path.join(UPLOADS_DIR, dataset.image_filename)
        try:
            os.remove(image_path)
        except FileNotFoundError:
            pass

    db.delete(dataset)
    db.commit()
    return {"message": f"Dataset '{dataset.filename}' deleted"}


@app.post("/start_inference/{model_id}")
async def start_inference_endpoint(model_id: str, db: Session = Depends(get_db)):
    global inference_processes

    if model_id in inference_processes and inference_processes[model_id].process.is_alive():
        return {
            "message": "Inference process already running",
            "status": "running",
            "model_id": model_id,
            "upload_url": f"http://localhost/inference/{model_id}",
            "json_url": "http://localhost:8000/docs#/default/predict_endpoint_predict__model_id__post"
        }

    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model or not db_model.model_filename:
        raise HTTPException(
            status_code=404,
            detail="Model not found or model data is missing"
        )

    try:
        model_path = os.path.join(MODELS_DIR, db_model.model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        result_queue = Queue()
        inference_process = InferenceProcess(
            model_id, db_model.model_filename, result_queue
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
        "upload_url": f"http://localhost/inference/{model_id}",
        "json_url": "http://localhost:8000/docs#/default/predict_endpoint_predict__model_id__post"
    }


@app.post("/predict/{model_id}")
async def predict_endpoint(model_id: str, data: List[Dict[str, Union[float, int, str]]], db: Session = Depends(get_db)):
    global inference_processes
    if model_id not in inference_processes or not inference_processes[model_id].process.is_alive():
        raise HTTPException(status_code=404, detail="Inference process not running or not found")

    result_queue = inference_processes[model_id].result_queue
    try:
        inference_processes[model_id].predict(data)
        result = result_queue.get(timeout=10)

        if result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"predictions": result["predictions"]}

    except multiprocessing.queues.Empty:
        raise HTTPException(status_code=500, detail="Prediction timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/stop_inference/{model_id}")
async def stop_inference_endpoint(model_id: str):
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
                            api_url=f"http://localhost/inference/{model_id}",
                        )
                    )
    return running_models_list

@app.post("/datasets/{dataset_id}/visualize", status_code=202) # 202 Accepted - задача принята
async def start_visualization_generation(dataset_id: str, db: Session = Depends(get_db)):
    """Запускает фоновую задачу генерации визуализаций для датасета."""
    logger.info(f"Received request to start visualization for dataset_id: {dataset_id}")
    # Проверка, существует ли датасет
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logger.warning(f"Dataset {dataset_id} not found for visualization request.")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Проверим, не запущена ли уже задача (можно добавить сложную логику, но пока просто запускаем)
    # Можно проверить статус существующей записи DatasetVisualization, если она PENDING

    # Запускаем Celery задачу
    task = generate_visualizations_task.delay(dataset_id)
    logger.info(f"Visualization task {task.id} queued for dataset_id: {dataset_id}")

    # Можно сразу создать запись в DatasetVisualization со статусом PENDING здесь,
    # чтобы /visualizations эндпоинт сразу ее видел. Или положиться на то, что
    # таска сама ее создаст/обновит. Таска надежнее, если она упадет до записи в БД.

    return {"task_id": task.id, "message": "Visualization task started."}


@app.get("/datasets/{dataset_id}/visualizations", response_model=Optional[VisualizationDataResponse])
async def get_visualizations(dataset_id: str, db: Session = Depends(get_db)):
    """Получает готовые данные для визуализаций или статус PENDING/FAILURE."""
    logger.debug(f"Fetching visualization data for dataset_id: {dataset_id}")
    visualization = db.query(DatasetVisualization)\
                      .filter(DatasetVisualization.dataset_id == dataset_id)\
                      .order_by(DatasetVisualization.generated_at.desc())\
                      .first() # Берем последнюю попытку

    if not visualization:
        logger.debug(f"No visualization record found for dataset_id: {dataset_id}")
        # Возвращаем None или пустой объект, или 404?
        # Пока вернем None, фронтенд поймет, что данных нет и надо запускать.
        # Если нужно различать "еще не запускали" и "не найдено", можно вернуть 404.
        # Но для логики "показать кнопку Visualize если нет данных" - None удобнее.
        return None
        # raise HTTPException(status_code=404, detail="Visualizations not generated yet or dataset not found.")


    logger.debug(f"Found visualization record for dataset_id {dataset_id} with status: {visualization.status}")
    # Используем Pydantic модель для формирования ответа
    return VisualizationDataResponse.from_orm(visualization)


@app.get("/visualization_status/{task_id}") # Можно использовать общий /train_status, но лучше отдельный
async def get_visualization_status(task_id: str):
    """Проверяет статус задачи генерации визуализаций Celery."""
    logger.debug(f"Checking status for visualization task_id: {task_id}")
    task_result = AsyncResult(task_id, app=celery) # Используем celery инстанс

    status = task_result.status
    result = task_result.result
    error_info = None

    if task_result.failed():
         status = "FAILURE" # Явно ставим FAILURE
         if isinstance(result, Exception):
             error_info = {
                'exc_type': type(result).__name__,
                'exc_message': str(result),
                'exc_traceback': getattr(task_result.backend, 'get_traceback', lambda x: "No traceback available.")(task_id)
            }
         elif isinstance(result, dict) and 'exc_type' in result: # Если таска вернула dict с ошибкой
              error_info = result
         else:
              error_info = {'exc_message': str(result) if result else "Unknown error"}

    elif task_result.successful():
        status = "SUCCESS" # Явно ставим SUCCESS

    response = {
        "task_id": task_id,
        "status": status, # PENDING, SUCCESS, FAILURE, RETRY, STARTED
        "result": result if task_result.successful() else None, # Результат только при успехе
        "error": error_info # Информация об ошибке при FAILURE
    }
    logger.debug(f"Status for task {task_id}: {response}")
    return response