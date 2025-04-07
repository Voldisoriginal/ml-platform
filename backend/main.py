import traceback
import logging
import json
import uuid
import os
import io
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Literal
from pydantic import BaseModel, validator, Field, ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Response, Query
import joblib
# ML
from sklearn.linear_model import LinearRegression, LogisticRegression # Добавим LogisticRegression для примера классификации
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier # Добавим DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, root_mean_squared_error # Добавим метрики классификации
# --- Новые импорты ---
import xgboost as xgb
import lightgbm as lgb
# --------------------
# Database (остается без изменений)
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Column, String, Float, JSON, LargeBinary, DateTime
from datetime import datetime
# Celery (остается без изменений)
from celery import Celery, states
from celery.exceptions import Ignore
# Multiprocessing (остается без изменений)
import multiprocessing
from multiprocessing import Process, Queue, Event
import numpy as np # Понадобится для гистограмм и корреляции
from scipy import stats # Может понадобиться для стат. деталей боксплотов
from celery.result import AsyncResult # Импортируем для проверки статуса

# --- Directory Setup (без изменений) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Database Configuration (без изменений) ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@db:5432/dbname")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models (без изменений - DBModel уже содержит model_type, params и т.д.) ---
class DBModel(Base):
    __tablename__ = "trained_models"
    id = Column(String, primary_key=True, index=True)
    model_type = Column(String) # Напр., 'DecisionTreeRegressor', 'XGBClassifier'
    task_type = Column(String) # 'regression' or 'classification' - добавим это поле!
    params = Column(JSON)
    metrics = Column(JSON)
    train_settings = Column(JSON)
    target_column = Column(String)
    dataset_filename = Column(String)
    model_filename = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)

class Dataset(Base): # без изменений
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

class DatasetVisualization(Base): # без изменений
    __tablename__ = "dataset_visualizations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, sqlalchemy.ForeignKey("datasets.id"), index=True, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    visualization_data = Column(JSON)
    status = Column(String, default="PENDING")
    error_message = Column(String, nullable=True)
    dataset = sqlalchemy.orm.relationship("Dataset")

# Create tables (если добавили поле task_type, может потребоваться миграция Alembic или ручное добавление колонки)
# Для простоты примера, предположим, что мы можем пересоздать таблицу или добавить колонку вручную:
# ALTER TABLE trained_models ADD COLUMN task_type VARCHAR;
try:
    Base.metadata.create_all(bind=engine)
    # Попытка добавить колонку, если её нет (не самый надежный способ для продакшена)
    with engine.connect() as connection:
        try:
            connection.execute(sqlalchemy.text("ALTER TABLE trained_models ADD COLUMN task_type VARCHAR;"))
            connection.commit() # Не забываем commit для DDL в SQLAlchemy 1.x/2.x без autocommit
        except sqlalchemy.exc.ProgrammingError as e:
            # Игнорируем ошибку, если колонка уже существует
            if "column \"task_type\" of relation \"trained_models\" already exists" in str(e):
                 pass # Колонка уже есть
            else:
                 raise # Другая ошибка
except Exception as e:
    print(f"Error during table creation or alteration: {e}")


# --- Celery Configuration (без изменений) ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", 'redis://redis:6379/0')
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

class TrainSettings(BaseModel): # без изменений
    train_size: float = 0.7
    random_state: int = 42
    @validator('train_size')
    def train_size_between_0_and_1(cls, v):
        if not 0 < v < 1:
            raise ValueError('train_size must be between 0 and 1')
        return v

# --- Новая структура для описания параметров модели ---
class ModelParameterDefinition(BaseModel):
    name: str # имя параметра как в библиотеке (e.g., 'n_estimators')
    label: str # Человекочитаемое имя (e.g., 'Number of Trees')
    type: Literal['integer', 'float', 'categorical', 'boolean'] # Тип данных
    default: Any # Значение по умолчанию
    component: Literal['InputNumber', 'Dropdown', 'Checkbox', 'InputText'] # Какой компонент PrimeVue использовать
    validation: Optional[Dict[str, Any]] = None # Правила валидации (e.g., {'min': 1, 'max': 1000} или {'values': ['gini', 'entropy']})

# --- Новая структура для описания доступной модели ---
class AvailableModelInfo(BaseModel):
    type: str # Уникальный идентификатор модели (e.g., 'XGBRegressor')
    name: str # Отображаемое имя (e.g., 'XGBoost Regressor')
    task_type: Literal['regression', 'classification'] # Тип задачи
    parameters: List[ModelParameterDefinition] # Список параметров для этой модели

# --- Обновленная ModelParams для валидации ---
class ModelParams(BaseModel):
    model_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    # Валидатор теперь будет проще, т.к. основную логику можно вынести
    # Или можно оставить детальную валидацию здесь, если нужно проверить комбинации и т.д.
    # Для примера, оставим базовую валидацию типов, которую можно расширить

    @validator("params", always=True) # always=True нужен, чтобы валидатор сработал даже для пустого dict
    def validate_params(cls, v, values):
        model_type = values.get("model_type")
        # Получаем определение модели (это нужно будет сделать доступным)
        # В реальном приложении лучше иметь сервис/менеджер для этого
        model_def = next((m for m in AVAILABLE_MODELS_LIST if m.type == model_type), None)
        if not model_def:
             raise ValueError(f"Unsupported model_type: {model_type}")

        validated_params = {}
        # Проверяем каждый переданный параметр
        for param_name, param_value in v.items():
            param_def = next((p for p in model_def.parameters if p.name == param_name), None)
            if not param_def:
                # Можно либо игнорировать неизвестные параметры, либо выдавать ошибку
                # raise ValueError(f"Unknown parameter '{param_name}' for model '{model_type}'")
                continue # Игнорируем для простоты

            # Проверка типа
            expected_type = param_def.type
            actual_type = type(param_value)

            valid_type = False
            if expected_type == 'integer' and isinstance(param_value, int):
                valid_type = True
            elif expected_type == 'float' and isinstance(param_value, (int, float)): # Разрешаем int для float полей
                param_value = float(param_value) # Приводим к float
                valid_type = True
            elif expected_type == 'boolean' and isinstance(param_value, bool):
                 valid_type = True
            elif expected_type == 'categorical' and isinstance(param_value, str): # Категориальные пока как строки
                 valid_type = True
            elif expected_type == 'text' and isinstance(param_value, str): # Добавим text для общих текстовых полей
                 valid_type = True


            if not valid_type:
                raise ValueError(f"Parameter '{param_name}' for model '{model_type}' expected type '{expected_type}', but got '{actual_type.__name__}'")

            # Проверка валидации (min, max, values)
            if param_def.validation:
                if 'min' in param_def.validation and param_value < param_def.validation['min']:
                    raise ValueError(f"Parameter '{param_name}' must be >= {param_def.validation['min']}")
                if 'max' in param_def.validation and param_value > param_def.validation['max']:
                    raise ValueError(f"Parameter '{param_name}' must be <= {param_def.validation['max']}")
                if 'values' in param_def.validation and param_value not in param_def.validation['values']:
                     allowed = ", ".join(map(str, param_def.validation['values']))
                     raise ValueError(f"Parameter '{param_name}' must be one of: {allowed}")

            validated_params[param_name] = param_value

        # Добавляем параметры по умолчанию, если они не были переданы
        for param_def in model_def.parameters:
             if param_def.name not in validated_params:
                 validated_params[param_def.name] = param_def.default

        return validated_params


class TrainedModel(BaseModel): # Добавим task_type
    id: str
    model_type: str
    task_type: Optional[str] = None # Allow task_type to be None
    params: Dict[str, Any]
    metrics: Dict[str, float]
    train_settings: TrainSettings # Используем Pydantic модель для вложенности
    target_column: str
    dataset_filename: str
    start_time: datetime

    class Config:
        orm_mode = True
        from_attributes = True

class DatasetModel(BaseModel): # без изменений
    id: str
    filename: str
    upload_date: datetime
    columns: List[str]
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    target_variable: Optional[str] = None
    imageUrl: Optional[str] = None
    file_size: Optional[int] = None
    row_count: Optional[int] = None
    column_types: Optional[Dict[str, str]] = None

    class Config:
        orm_mode = True
        from_attributes = True

class DatasetPreviewResponse(BaseModel): # без изменений
    preview: List[Dict[str, Any]]
    column_types: Optional[Dict[str, str]] = None

class RunningModel(BaseModel): # без изменений
    model_id: str
    dataset_filename: str
    target_column: str
    model_type: str
    metrics: Dict[str, float]
    status: str
    api_url: str

class VisualizationDataResponse(BaseModel): # без изменений
    id: str
    dataset_id: str
    generated_at: datetime
    visualization_data: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str] = None
    class Config:
        orm_mode = True
        from_attributes = True

# --- Определение доступных моделей (Глобальная переменная) ---
# Это можно вынести в отдельный конфигурационный файл или базу данных
AVAILABLE_MODELS_LIST: List[AvailableModelInfo] = [
    # --- Regression ---
    AvailableModelInfo(
        type='LinearRegression', name='Linear Regression', task_type='regression',
        parameters=[] # У линейной регрессии в sklearn нет критичных гиперпараметров для базовой настройки
    ),
    AvailableModelInfo(
        type='DecisionTreeRegressor', name='Decision Tree Regressor', task_type='regression',
        parameters=[
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=None, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='min_samples_split', label='Min Samples Split', type='integer', default=2, component='InputNumber', validation={'min': 2}),
            ModelParameterDefinition(name='min_samples_leaf', label='Min Samples Leaf', type='integer', default=1, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='criterion', label='Criterion', type='categorical', default='squared_error', component='Dropdown', validation={'values': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}),
        ]
    ),
    AvailableModelInfo(
        type='XGBRegressor', name='XGBoost Regressor', task_type='regression',
        parameters=[
            ModelParameterDefinition(name='n_estimators', label='N Estimators', type='integer', default=100, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='learning_rate', label='Learning Rate', type='float', default=0.1, component='InputNumber', validation={'min': 0.0001, 'max': 1.0}),
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=3, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='subsample', label='Subsample', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='colsample_bytree', label='Colsample Bytree', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
             # Добавим objective, хотя часто он по умолчанию подходит
             ModelParameterDefinition(name='objective', label='Objective', type='categorical', default='reg:squarederror', component='Dropdown', validation={'values': ['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror', 'reg:absoluteerror']}),
        ]
    ),
    AvailableModelInfo(
        type='LGBMRegressor', name='LightGBM Regressor', task_type='regression',
        parameters=[
            ModelParameterDefinition(name='n_estimators', label='N Estimators', type='integer', default=100, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='learning_rate', label='Learning Rate', type='float', default=0.1, component='InputNumber', validation={'min': 0.0001, 'max': 1.0}),
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=-1, component='InputNumber', validation={'min': -1}), # -1 means no limit in LightGBM
            ModelParameterDefinition(name='num_leaves', label='Num Leaves', type='integer', default=31, component='InputNumber', validation={'min': 2}),
            ModelParameterDefinition(name='subsample', label='Subsample (Bagging Fraction)', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='colsample_bytree', label='Colsample Bytree (Feature Fraction)', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
             ModelParameterDefinition(name='objective', label='Objective', type='categorical', default='regression', component='Dropdown', validation={'values': ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie']}),
        ]
    ),
    # --- Classification ---
     AvailableModelInfo(
        type='LogisticRegression', name='Logistic Regression', task_type='classification',
        parameters=[
            ModelParameterDefinition(name='penalty', label='Penalty', type='categorical', default='l2', component='Dropdown', validation={'values': ['l1', 'l2', 'elasticnet', None]}), # None может потребовать solver='saga'
            ModelParameterDefinition(name='C', label='C (Inverse Regularization)', type='float', default=1.0, component='InputNumber', validation={'min': 0.0001}),
            ModelParameterDefinition(name='solver', label='Solver', type='categorical', default='lbfgs', component='Dropdown', validation={'values': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}),
             ModelParameterDefinition(name='max_iter', label='Max Iterations', type='integer', default=100, component='InputNumber', validation={'min': 10}),
        ]
    ),
    AvailableModelInfo(
        type='DecisionTreeClassifier', name='Decision Tree Classifier', task_type='classification',
        parameters=[
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=None, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='min_samples_split', label='Min Samples Split', type='integer', default=2, component='InputNumber', validation={'min': 2}),
            ModelParameterDefinition(name='min_samples_leaf', label='Min Samples Leaf', type='integer', default=1, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='criterion', label='Criterion', type='categorical', default='gini', component='Dropdown', validation={'values': ['gini', 'entropy', 'log_loss']}),
        ]
    ),
    AvailableModelInfo(
        type='XGBClassifier', name='XGBoost Classifier', task_type='classification',
        parameters=[ # Параметры схожи с регрессором, но objective другой
            ModelParameterDefinition(name='n_estimators', label='N Estimators', type='integer', default=100, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='learning_rate', label='Learning Rate', type='float', default=0.1, component='InputNumber', validation={'min': 0.0001, 'max': 1.0}),
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=3, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='subsample', label='Subsample', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='colsample_bytree', label='Colsample Bytree', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='objective', label='Objective', type='categorical', default='binary:logistic', component='Dropdown', validation={'values': ['binary:logistic', 'binary:logitraw', 'binary:hinge', 'multi:softmax', 'multi:softprob']}), # Добавили multi-class
            ModelParameterDefinition(name='eval_metric', label='Eval Metric', type='categorical', default='logloss', component='Dropdown', validation={'values': ['rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg', 'map']}), # Добавили метрики оценки
        ]
    ),
        AvailableModelInfo(
        type='LGBMClassifier', name='LightGBM Classifier', task_type='classification',
        parameters=[ # Параметры схожи с регрессором, но objective другой
            ModelParameterDefinition(name='n_estimators', label='N Estimators', type='integer', default=100, component='InputNumber', validation={'min': 1}),
            ModelParameterDefinition(name='learning_rate', label='Learning Rate', type='float', default=0.1, component='InputNumber', validation={'min': 0.0001, 'max': 1.0}),
            ModelParameterDefinition(name='max_depth', label='Max Depth', type='integer', default=-1, component='InputNumber', validation={'min': -1}),
            ModelParameterDefinition(name='num_leaves', label='Num Leaves', type='integer', default=31, component='InputNumber', validation={'min': 2}),
            ModelParameterDefinition(name='subsample', label='Subsample (Bagging Fraction)', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='colsample_bytree', label='Colsample Bytree (Feature Fraction)', type='float', default=1.0, component='InputNumber', validation={'min': 0.1, 'max': 1.0}),
            ModelParameterDefinition(name='objective', label='Objective', type='categorical', default='binary', component='Dropdown', validation={'values': ['binary', 'multiclass', 'ovr']}), # Добавили multiclass
            # LightGBM часто сам определяет метрику, но можно указать
            ModelParameterDefinition(name='metric', label='Metric', type='categorical', default='binary_logloss', component='Dropdown', validation={'values': ['binary_logloss', 'auc', 'average_precision', 'binary_error', 'multi_logloss', 'multi_error']}),
        ]
    ),
]


# --- FastAPI App ---
app = FastAPI()

# --- CORS (без изменений) ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Helpers (без изменений) ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Logging Setup (без изменений) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- Celery Tasks ---

@celery.task(bind=True, name='train_model_task')
def train_model_task(self, dataset_filename: str, target_column: str,
                     train_settings: Dict, model_params: Dict):
    try:
        # 1. Load dataset (без изменений)
        logger.debug(f"Loading dataset: {dataset_filename}")
        dataset_path = os.path.join(UPLOADS_DIR, dataset_filename)
        df = pd.read_csv(dataset_path) # Предполагаем, что файл существует (проверки опущены для краткости)

        if target_column not in df.columns:
            # Обработка ошибки (без изменений)
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # --- Определяем тип задачи (регрессия или классификация) ---
        # Простой способ: проверить тип данных целевой колонки
        target_dtype = df[target_column].dtype
        if pd.api.types.is_numeric_dtype(target_dtype) and df[target_column].nunique() > 15: # Эвристика: много уникальных числовых -> регрессия
             task_type = 'regression'
             logger.info(f"Detected task type: Regression (target='{target_column}', dtype={target_dtype})")
        else: # Иначе считаем классификацией (включая строки, bool, числа с малым количеством уникальных значений)
             task_type = 'classification'
             logger.info(f"Detected task type: Classification (target='{target_column}', dtype={target_dtype})")
             # Для классификации может потребоваться кодирование целевой переменной, если она строковая
             if pd.api.types.is_string_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype):
                 logger.warning(f"Target column '{target_column}' is of type {target_dtype}. Attempting label encoding.")
                 # TODO: Implement robust label encoding if needed
                 try:
                    df[target_column] = df[target_column].astype('category').cat.codes
                    logger.info(f"Target column '{target_column}' label encoded.")
                 except Exception as e:
                     logger.error(f"Failed to label encode target column '{target_column}': {e}")
                     raise ValueError(f"Could not encode target column '{target_column}' for classification.")


        # TODO: Обработка категориальных фичей (OneHotEncoding, etc.) - ВАЖНО для многих моделей!
        # Простой пример: OneHotEncode все object/category колонки, кроме целевой
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_features:
             categorical_features.remove(target_column)

        if categorical_features:
             logger.info(f"Applying OneHotEncoding to features: {categorical_features}")
             df = pd.get_dummies(df, columns=categorical_features, drop_first=True) # drop_first=True чтобы избежать мультиколлинеарности
             logger.info(f"DataFrame shape after encoding: {df.shape}")


        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # --- Проверка совместимости модели и задачи ---
        model_info = next((m for m in AVAILABLE_MODELS_LIST if m.type == model_params['model_type']), None)
        if not model_info:
             raise ValueError(f"Model type {model_params['model_type']} definition not found.")
        if model_info.task_type != task_type:
            raise ValueError(f"Model {model_params['model_type']} is for {model_info.task_type}, but detected task is {task_type}.")


        # 2. Transform train settings (без изменений)
        logger.debug(f"Transforming train settings: {train_settings}")
        train_settings_obj = TrainSettings(**train_settings)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_settings_obj.train_size,
            random_state=train_settings_obj.random_state,
            stratify=y if task_type == 'classification' else None # Стратификация для классификации
        )

        # 3. Train model (Обновлено для новых моделей)
        logger.debug(f"Validating and processing model parameters: {model_params}")
        # Валидация уже должна была произойти в эндпоинте, но можно перепроверить
        try:
             model_params_obj = ModelParams(**model_params) # Валидируем и получаем параметры с дефолтами
        except ValidationError as e:
             logger.error(f"Model parameter validation failed: {e}")
             raise e # Пробрасываем ошибку

        model_type = model_params_obj.model_type
        params = model_params_obj.params
        logger.info(f"Initializing model: {model_type} with params: {params}")

        # --- Инициализация модели ---
        if model_type == "LinearRegression":
            model = LinearRegression(**params)
        elif model_type == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(**params, random_state=train_settings_obj.random_state) # Добавим random_state
        elif model_type == "XGBRegressor":
            model = xgb.XGBRegressor(**params, random_state=train_settings_obj.random_state)
        elif model_type == "LGBMRegressor":
             model = lgb.LGBMRegressor(**params, random_state=train_settings_obj.random_state)
        # --- Классификация ---
        elif model_type == "LogisticRegression":
             model = LogisticRegression(**params, random_state=train_settings_obj.random_state)
        elif model_type == "DecisionTreeClassifier":
             model = DecisionTreeClassifier(**params, random_state=train_settings_obj.random_state)
        elif model_type == "XGBClassifier":
             # XGBoost может требовать кодирования классов от 0 до N-1
             model = xgb.XGBClassifier(**params, random_state=train_settings_obj.random_state, use_label_encoder=False, eval_metric=params.get('eval_metric', 'logloss')) # use_label_encoder=False рекомендуется
        elif model_type == "LGBMClassifier":
             model = lgb.LGBMClassifier(**params, random_state=train_settings_obj.random_state)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError("Unsupported model type")

        logger.debug("Training model...")
        model.fit(X_train, y_train)
        logger.debug("Training complete.")

        # 4. Evaluate model (Обновлено для разных задач)
        y_pred = model.predict(X_test)
        metrics = {}
        if task_type == 'regression':
            metrics["r2_score"] = r2_score(y_test, y_pred)
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = root_mean_squared_error(y_test, y_pred) # Добавим RMSE
        elif task_type == 'classification':
            # Для бинарной и мультиклассовой классификации
            metrics["accuracy_score"] = accuracy_score(y_test, y_pred)
            # F1-score может требовать указания average для мультиклассовой
            avg_method = 'weighted' if y.nunique() > 2 else 'binary'
            try:
                 metrics["f1_score"] = f1_score(y_test, y_pred, average=avg_method)
            except ValueError as e:
                 logger.warning(f"Could not calculate F1 score (possibly due to labels): {e}")
                 metrics["f1_score"] = None # Или 0, или пропустить

            # TODO: Добавить другие метрики классификации (precision, recall, AUC)
            # Для AUC может понадобиться predict_proba

        logger.debug(f"Metrics: {metrics}")
        model_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Save the model to a file (без изменений)
        model_filename = f"{model_id}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)

        # Save to database (Обновлено - добавляем task_type)
        logger.debug("Saving to database...")
        with SessionLocal() as db:
            db_model = DBModel(
                id=model_id,
                model_type=model_type,
                task_type=task_type, # Сохраняем тип задачи
                params=params, # Сохраняем фактически использованные параметры (включая дефолты)
                metrics=metrics,
                train_settings=train_settings, # Сохраняем исходные настройки
                target_column=target_column,
                dataset_filename=dataset_filename,
                model_filename=model_filename,
                start_time=start_time
            )
            db.add(db_model)
            db.commit()
            # Создаем объект Pydantic для возврата, используя данные из БД
            result = TrainedModel.from_orm(db_model).dict() # Используем Pydantic модель
            # Convert datetime to string for JSON serialization if necessary
            result['start_time'] = result['start_time'].isoformat()


        logger.debug("Database save complete.")
        return result

    except (ValidationError, ValueError) as e: # Ловим ошибки валидации и ValueError отдельно
        logger.error(f"Validation or Value Error in Celery task: {e}", exc_info=True)
        self.update_state(
            state=states.FAILURE,
            meta={ 'exc_type': type(e).__name__, 'exc_message': str(e), 'traceback': traceback.format_exc()}
        )
        raise Ignore()
    except Exception as e:
        logger.exception("An unexpected error occurred in the Celery task:")
        self.update_state(
            state=states.FAILURE,
            meta={ 'exc_type': type(e).__name__, 'exc_message': str(e), 'traceback': traceback.format_exc()}
        )
        raise Ignore()


# generate_visualizations_task (без изменений)
@celery.task(bind=True, name='generate_visualizations_task')
def generate_visualizations_task(self, dataset_id: str):
    # ... (код задачи без изменений) ...
    pass # Убрал код для краткости, он остается прежним

# InferenceProcess class (без изменений)
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
                # break

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

# --- Новый эндпоинт для получения списка моделей ---
@app.get("/available_models/", response_model=List[AvailableModelInfo])
async def get_available_models():
    """Возвращает список доступных моделей с их параметрами."""
    return AVAILABLE_MODELS_LIST

# --- Существующие эндпоинты (без изменений, кроме train и get_trained_models) ---

@app.post("/upload_dataset/") # без изменений
async def upload_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        description: Optional[str] = Form(None),
        author: Optional[str] = Form(None),
        target_variable: Optional[str] = Form(None),
        image: Optional[UploadFile] = Form(None),
        db: Session = Depends(get_db)
):
    # ... (код без изменений) ...
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type {file.content_type}")

    try:
        if not file.filename.lower().endswith(".csv"): # Проверка по расширению надежнее
             raise HTTPException(
                 status_code=400, detail="Invalid file type. Must be CSV."
             )

        contents = await file.read()
        # Попытка определить кодировку (опционально, но полезно)
        try:
            decoded_contents = contents.decode('utf-8')
        except UnicodeDecodeError:
            try:
                 decoded_contents = contents.decode('cp1251') # Попробовать другую популярную кодировку
                 logger.warning(f"Decoded {file.filename} using cp1251.")
            except UnicodeDecodeError:
                 logger.error(f"Could not decode {file.filename} with utf-8 or cp1251.")
                 raise HTTPException(status_code=400, detail="Could not decode file. Ensure it's UTF-8 or CP1251 encoded CSV.")

        # Проверка на пустой файл или только заголовок
        if not decoded_contents.strip() or len(decoded_contents.strip().splitlines()) < 2:
            raise HTTPException(status_code=400, detail="CSV file is empty or contains only a header.")

        df = pd.read_csv(io.StringIO(decoded_contents))

        if df.empty:
             raise HTTPException(status_code=400, detail="CSV file loaded as empty DataFrame.")

        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOADS_DIR, filename)

        # Проверка существования датасета по имени, данному пользователем (name)
        existing_dataset_by_name = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset_by_name:
             raise HTTPException(status_code=409, detail=f"Dataset with name '{name}' already exists.")

        # Запись файла на диск
        with open(filepath, "wb") as f:
            f.write(contents) # Записываем оригинальные байты

        image_filename = None
        if image:
            if not image.content_type.startswith("image/"):
                # Удаляем загруженный csv, если картинка невалидна
                os.remove(filepath)
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
        # Если файл был создан, но произошла ошибка (напр., с картинкой или БД), удалить файл
        if 'filepath' in locals() and os.path.exists(filepath):
             os.remove(filepath)
        if 'image_path' in locals() and os.path.exists(image_path):
             os.remove(image_path)
        raise e
    except pd.errors.ParserError as e:
         logger.error(f"Error parsing CSV file {file.filename}: {e}")
         raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {e}")
    except Exception as e:
        logger.exception("Error uploading dataset")
        # Попытка удалить файлы, если они были созданы
        if 'filepath' in locals() and os.path.exists(filepath):
             os.remove(filepath)
        if 'image_path' in locals() and os.path.exists(image_path):
             os.remove(image_path)
        db.rollback() # Откатить транзакцию БД
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/dataset/{dataset_id}", response_model=DatasetModel) # без изменений
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
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
            # Улучшенная версия для разных кодировок
            row_count = 0
            try:
                 with open(filepath, 'r', encoding='utf-8') as f:
                      row_count = sum(1 for _ in f) - 1 # Subtract 1 for header
            except UnicodeDecodeError:
                 try:
                     with open(filepath, 'r', encoding='cp1251') as f:
                          row_count = sum(1 for _ in f) - 1
                     logger.warning(f"Counted rows using cp1251 for {dataset.filename}")
                 except Exception as count_err:
                     logger.error(f"Could not count rows for {dataset.filename}: {count_err}")
                     row_count = -3 # Indicate error counting rows

            logger.debug(f"Row count for {dataset.filename}: {row_count}")

            # Get column types using pandas on a sample
            # Handle potential decoding errors here too
            df_sample = None
            try:
                 df_sample = pd.read_csv(filepath, nrows=50, encoding='utf-8')
            except UnicodeDecodeError:
                 try:
                     df_sample = pd.read_csv(filepath, nrows=50, encoding='cp1251')
                 except Exception as pd_err:
                     logger.error(f"Could not read sample to get types for {dataset.filename}: {pd_err}")

            if df_sample is not None:
                 column_types = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                 logger.debug(f"Column types detected for {dataset.filename}: {column_types}")
            else:
                 column_types = {"error": "Could not read sample"}


        except FileNotFoundError:
             logger.error(f"File not found on disk for dataset {dataset_id} at path {filepath}, though DB record exists.")
             file_size = -1
             row_count = -1
        except Exception as e:
            logger.error(f"Error getting file stats for {dataset.filename} (ID: {dataset_id}): {e}", exc_info=True)
            file_size = -2
            row_count = -2

    # Construct the response model using all gathered info
    response_data = DatasetModel(
        id=dataset.id,
        filename=dataset.filename,
        upload_date=dataset.upload_date,
        columns=dataset.columns,
        name=dataset.name,
        description=dataset.description,
        author=dataset.author,
        target_variable=dataset.target_variable,
        imageUrl=image_url,
        file_size=file_size,
        row_count=row_count,
        column_types=column_types
    )
    logger.debug(f"Returning details for dataset ID: {dataset_id}")
    return response_data


@app.get("/dataset/{dataset_id}/image") # без изменений
async def get_dataset_image(dataset_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset or not dataset.image_filename:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = os.path.join(UPLOADS_DIR, dataset.image_filename)
    if not os.path.exists(image_path):
        # Попытка найти placeholder, если реальное изображение отсутствует
        placeholder_path = os.path.join(BASE_DIR, "placeholder.png")
        if os.path.exists(placeholder_path):
             logger.warning(f"Serving placeholder for missing image: {dataset.image_filename}")
             with open(placeholder_path, "rb") as image_file:
                 return Response(content=image_file.read(), media_type="image/png")
        else:
             raise HTTPException(status_code=404, detail="Image file not found on server and no placeholder available")

    try:
        # Определяем media type по расширению файла (простой вариант)
        media_type = "image/jpeg" # Default
        ext = os.path.splitext(dataset.image_filename)[1].lower()
        if ext == ".png":
            media_type = "image/png"
        elif ext == ".gif":
            media_type = "image/gif"
        elif ext == ".webp":
            media_type = "image/webp"

        with open(image_path, "rb") as image_file:
            return Response(content=image_file.read(), media_type=media_type)
    except Exception as e:
        logger.exception(f"Error reading image file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading image file: {e}")


@app.get("/datasets/", response_model=List[DatasetModel]) # без изменений
async def list_datasets(db: Session = Depends(get_db)):
     # ... (код без изменений) ...
    logger.debug("Fetching list of all datasets.")
    datasets_db = db.query(Dataset).order_by(Dataset.upload_date.desc()).all() # Order by upload date
    response_data = []
    for dataset in datasets_db:
        image_url = f"/dataset/{dataset.id}/image" if dataset.image_filename else "/placeholder.png" # Ссылка на плейсхолдер по умолчанию
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
             file_size=None,
             row_count=None,
             column_types=None
         )
        response_data.append(dataset_model)

    logger.debug(f"Returning {len(response_data)} datasets.")
    return response_data


@app.get("/dataset_preview/{filename}", response_model=DatasetPreviewResponse) # без изменений
async def get_dataset_preview(
    filename: str,
    rows: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
    ):
     # ... (код без изменений) ...
    logger.debug(f"Request for preview of '{filename}', rows={rows}")
    filepath = os.path.join(UPLOADS_DIR, filename)
    actual_filename = filename
    dataset_id_used = None

    # Check if the file exists directly
    if not os.path.exists(filepath):
        logger.warning(f"Preview request: File '{filename}' not found directly. Checking if it's a dataset ID.")
        # If not found, check if 'filename' might be a dataset ID
        dataset_by_id = db.query(Dataset).filter(Dataset.id == filename).first()
        if dataset_by_id and dataset_by_id.filename:
            dataset_id_used = filename # Запомним, что использовали ID
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
        # Пытаемся угадать кодировку при чтении превью
        df_preview = None
        try:
             df_preview = pd.read_csv(filepath, nrows=rows, encoding='utf-8')
        except UnicodeDecodeError:
             try:
                  df_preview = pd.read_csv(filepath, nrows=rows, encoding='cp1251')
                  logger.warning(f"Read preview using cp1251 for {actual_filename}")
             except Exception as pd_err:
                  logger.error(f"Could not read preview for {actual_filename}: {pd_err}")
                  raise HTTPException(status_code=500, detail=f"Could not read preview from file: {pd_err}")
        except pd.errors.ParserError as e:
             logger.error(f"Parser error reading preview for {actual_filename}: {e}")
             raise HTTPException(status_code=400, detail=f"Error parsing CSV file for preview: {e}")


        if df_preview is None: # Если не удалось прочитать
             raise HTTPException(status_code=500, detail="Failed to read dataset preview.")


        # Convert NaN/NaT to None for JSON compatibility
        df_cleaned = df_preview.where(pd.notnull(df_preview), None)
        preview_data = df_cleaned.to_dict(orient='records')

        # Get column types from the preview
        column_types = {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
        logger.debug(f"Preview generated successfully for {actual_filename}.")
        return DatasetPreviewResponse(preview=preview_data, column_types=column_types)

    except FileNotFoundError:
         logger.error(f"FileNotFoundError during preview generation for {actual_filename} at {filepath}.")
         raise HTTPException(status_code=404, detail="Dataset file not found during preview generation.")
    except pd.errors.EmptyDataError:
         logger.warning(f"EmptyDataError during preview for {actual_filename}. Returning empty preview.")
         return DatasetPreviewResponse(preview=[], column_types={})
    except HTTPException as e:
        raise e # Пробрасываем HTTP исключения
    except Exception as e:
        logger.error(f"Error reading dataset preview for {actual_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading dataset preview: {e}")


@app.get("/placeholder.png") # без изменений
async def get_placeholder_image():
    # ... (код без изменений) ...
    placeholder_path = os.path.join(BASE_DIR, "placeholder.png") # Assuming it's in the same directory as backend.py
    fallback_placeholder_path = os.path.join(os.path.dirname(UPLOADS_DIR), "placeholder.png") # Or maybe one level up from uploads

    final_path_to_use = None
    if os.path.exists(placeholder_path):
        final_path_to_use = placeholder_path
    elif os.path.exists(fallback_placeholder_path):
        final_path_to_use = fallback_placeholder_path

    if not final_path_to_use:
        # Create a very basic placeholder image on-the-fly if needed ONLY if it doesn't exist
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (300, 200), color=(200, 200, 200))
            d = ImageDraw.Draw(img)
            d.text((90, 90), "Placeholder", fill=(50, 50, 50)) # Centered text roughly
            # Пытаемся сохранить в директорию BASE_DIR
            target_save_path = placeholder_path # По умолчанию сохраняем рядом с backend.py
            try:
                 img.save(target_save_path)
                 final_path_to_use = target_save_path
                 logger.info(f"Created and saved placeholder image at: {final_path_to_use}")
            except IOError as save_err:
                 logger.error(f"Could not save generated placeholder image at {target_save_path}: {save_err}")
                 # Если не удалось сохранить, вернем 404 или ошибку сервера
                 raise HTTPException(status_code=500, detail="Could not create or serve placeholder image.")

        except ImportError:
            logger.error("Pillow library not installed. Cannot generate placeholder image.")
            raise HTTPException(status_code=501, detail="Image generation library not available.") # 501 Not Implemented
        except Exception as gen_err:
            logger.error(f"Error generating placeholder image: {gen_err}")
            raise HTTPException(status_code=500, detail="Error generating placeholder image.")


    # Если дошли сюда, final_path_to_use должен быть валидным путем
    try:
        with open(final_path_to_use, "rb") as image_file:
            return Response(content=image_file.read(), media_type="image/png")
    except Exception as read_err:
         logger.error(f"Error reading placeholder file at {final_path_to_use}: {read_err}")
         raise HTTPException(status_code=500, detail="Error reading placeholder file.")


# Обновленный эндпоинт train
@app.post("/train/")
async def train_model(
        dataset_filename: str = Form(...),
        target_column: str = Form(...),
        train_settings: str = Form(...), # JSON string
        model_params: str = Form(...), # JSON string { "model_type": "...", "params": {...} }
        db: Session = Depends(get_db) # Добавим зависимость от БД для проверки датасета
):
    logger.info(f"Received training request for dataset: {dataset_filename}, target: {target_column}")
    logger.debug(f"Train settings (raw): {train_settings}")
    logger.debug(f"Model params (raw): {model_params}")

    # 1. Проверка существования датасета
    dataset_path = os.path.join(UPLOADS_DIR, dataset_filename)
    if not os.path.exists(dataset_path):
         # Дополнительно проверим в БД
         dataset_db = db.query(Dataset).filter(Dataset.filename == dataset_filename).first()
         if not dataset_db:
             logger.error(f"Dataset file or DB record not found for filename: {dataset_filename}")
             raise HTTPException(status_code=404, detail=f"Dataset '{dataset_filename}' not found.")
         else:
             logger.error(f"Dataset DB record found but file missing on disk: {dataset_path}")
             raise HTTPException(status_code=404, detail=f"Dataset file '{dataset_filename}' not found on server.")


    # 2. Валидация входных JSON
    try:
        train_settings_dict = json.loads(train_settings)
        model_params_dict = json.loads(model_params) # Ожидаем {"model_type": "...", "params": {...}}

        # Базовая проверка структуры
        if not isinstance(train_settings_dict, dict):
            raise TypeError("train_settings must be a JSON object")
        if not isinstance(model_params_dict, dict) or "model_type" not in model_params_dict or "params" not in model_params_dict:
             raise TypeError("model_params must be a JSON object with 'model_type' and 'params' keys")
        if not isinstance(model_params_dict["params"], dict):
             raise TypeError("model_params.params must be a JSON object")

        # Валидация с помощью Pydantic
        validated_train_settings = TrainSettings(**train_settings_dict).dict() # Получаем dict после валидации
        validated_model_params = ModelParams(**model_params_dict).dict() # Получаем dict после валидации

        logger.debug(f"Validated Train Settings: {validated_train_settings}")
        logger.debug(f"Validated Model Params: {validated_model_params}")

    except (json.JSONDecodeError, TypeError, ValidationError) as e:
        logger.error(f"Invalid input format or validation error: {e}", exc_info=True)
        # Формируем более понятное сообщение об ошибке для фронтенда
        error_detail = f"Invalid input: {e}"
        if isinstance(e, ValidationError):
             # Можно извлечь более конкретные ошибки из e.errors()
             try:
                 first_error = e.errors()[0]
                 loc = " -> ".join(map(str, first_error['loc'])) if first_error.get('loc') else 'input'
                 error_detail = f"Validation Error at '{loc}': {first_error['msg']}"
             except Exception:
                 error_detail = f"Validation Error: {e}" # Fallback

        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e: # Ловим другие возможные ошибки при парсинге
         logger.error(f"Unexpected error during input processing: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal error processing input: {e}")


    # 3. Запуск задачи Celery
    try:
        task = train_model_task.delay(
            dataset_filename,
            target_column,
            validated_train_settings, # Передаем провалидированный dict
            validated_model_params  # Передаем провалидированный dict
        )
        logger.info(f"Training task {task.id} queued for model {validated_model_params['model_type']}")
        return {"task_id": task.id}
    except Exception as e:
         logger.error(f"Failed to queue Celery task: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to start training task.")


@app.get("/train_status/{task_id}") # без изменений
async def get_train_status(task_id: str):
    # ... (код без изменений) ...
    task_result = AsyncResult(task_id, app=celery) # Убедимся, что используем наш celery instance
    result_data = task_result.result
    status = task_result.status
    error_info = None

    # Дополнительная обработка результата и статуса
    if task_result.failed():
        status = "FAILURE" # Убедимся, что статус FAILURE
        if isinstance(result_data, dict) and 'exc_type' in result_data and 'exc_message' in result_data:
            # Если задача вернула dict с информацией об ошибке (как мы делаем в task)
            error_info = {
                'type': result_data.get('exc_type', 'UnknownError'),
                'message': result_data.get('exc_message', 'No message'),
                'traceback': result_data.get('traceback', None) # traceBACK а не tracefoRWard :)
            }
        elif isinstance(result_data, Exception):
             # Если Celery сам перехватил исключение
             error_info = {
                 'type': type(result_data).__name__,
                 'message': str(result_data),
                 'traceback': getattr(task_result, 'traceback', None) # Попытка получить traceback из результата
             }
             # Попытка получить traceback через backend, если доступно
             if not error_info['traceback'] and hasattr(task_result.backend, 'get_task_meta'):
                  try:
                      meta = task_result.backend.get_task_meta(task_id)
                      error_info['traceback'] = meta.get('traceback')
                  except Exception:
                      pass # Не удалось получить метаданные
        else:
             # Неизвестный формат ошибки
             error_info = {
                 'type': "UnknownError",
                 'message': str(result_data) if result_data else "Unknown failure reason",
                 'traceback': getattr(task_result, 'traceback', None)
             }
        # Очищаем result, так как он содержит информацию об ошибке
        result_data = None

    elif task_result.successful():
        status = "SUCCESS" # Убедимся, что статус SUCCESS

    response = {
        "task_id": task_id,
        "status": status, # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
        "result": result_data, # Данные только при SUCCESS
        "error": error_info, # Информация об ошибке при FAILURE
    }

    # Опционально: Логирование статуса запроса
    # logger.debug(f"Status check for task {task_id}: {response['status']}")
    # if response['error']:
    #     logger.warning(f"Task {task_id} failed: {response['error']}")

    return response


# Обновленный эндпоинт для получения списка обученных моделей
# Используем модель TrainedModel, которая теперь включает task_type
@app.get("/trained_models/", response_model=List[TrainedModel])
async def get_trained_models(db: Session = Depends(get_db)):
    db_models = db.query(DBModel).order_by(DBModel.start_time.desc()).all()
    # Используем Pydantic модель TrainedModel для валидации и преобразования
    return [TrainedModel.from_orm(db_model) for db_model in db_models]


@app.get("/trained_models/search_sort", response_model=List[TrainedModel]) # Используем обновленную TrainedModel
async def search_sort_trained_models(
        db: Session = Depends(get_db),
        search_query: Optional[str] = Query(None, description="Search by ID, dataset filename, or target column"),
        sort_by: Optional[str] = Query("start_time", description="Column to sort by (id, model_type, task_type, start_time, r2_score, mse, accuracy_score, f1_score)"),
        sort_order: Optional[Literal['asc', 'desc']] = Query("desc", description="Sort order"),
        model_type: Optional[str] = Query(None, description="Filter by specific model type (e.g., XGBRegressor)"),
        task_type: Optional[Literal['regression', 'classification']] = Query(None, description="Filter by task type"), # Добавлен фильтр по типу задачи
        dataset_filename: Optional[str] = Query(None, description="Filter by dataset filename (exact match or partial with wildcard)"),
        model_id: Optional[str] = Query(None, description="Filter by specific model ID") # Добавляем новый параметр
):
    logger.debug(f"Search/Sort request: query='{search_query}', sort='{sort_by}' ({sort_order}), model='{model_type}', task='{task_type}', dataset='{dataset_filename}', id='{model_id}'")
    query = db.query(DBModel)

    # --- Фильтрация ---
    if model_id:
        query = query.filter(DBModel.id == model_id)
    # Важно: Если указан ID, остальные фильтры поиска/текста игнорируем? Или комбинируем?
    # Пока комбинируем, но поиск по ID должен вернуть максимум 1 результат.

    if search_query:
         # Используем ilike для регистронезависимого поиска
         search_pattern = f"%{search_query}%"
         query = query.filter(
             sqlalchemy.or_(
                 DBModel.id.ilike(search_pattern),
                 DBModel.dataset_filename.ilike(search_pattern),
                 DBModel.target_column.ilike(search_pattern),
                 DBModel.model_type.ilike(search_pattern) # Добавим поиск по типу модели
             )
         )

    if model_type:
        query = query.filter(DBModel.model_type == model_type) # Точное совпадение типа модели

    if task_type: # Новый фильтр
        query = query.filter(DBModel.task_type == task_type)

    if dataset_filename:
        # Позволим использовать wildcard * или % в запросе
        if "*" in dataset_filename or "%" in dataset_filename:
             search_pattern_ds = dataset_filename.replace("*", "%")
             query = query.filter(DBModel.dataset_filename.ilike(search_pattern_ds))
        else:
             # Если нет wildcard, ищем точное совпадение (или ilike для регистронезависимости?)
             # query = query.filter(DBModel.dataset_filename == dataset_filename)
             query = query.filter(DBModel.dataset_filename.ilike(dataset_filename)) # Оставим ilike

    # --- Сортировка ---
    sort_column = None
    order_func = sqlalchemy.asc if sort_order == "asc" else sqlalchemy.desc

    # Сортировка по полям модели
    simple_sort_columns = ["id", "model_type", "task_type", "start_time", "target_column", "dataset_filename"]
    if sort_by in simple_sort_columns:
        sort_column = getattr(DBModel, sort_by, None)
        if sort_column:
             query = query.order_by(order_func(sort_column))

    # Сортировка по метрикам (из JSON поля metrics)
    elif sort_by in ["r2_score", "mse", "rmse", "accuracy_score", "f1_score"]:
        # Используем ->> для извлечения значения как текста, затем cast в Float
        # Обрабатываем NULL значения (например, если метрика не применима)
        # PostgreSQL: NULLS LAST / NULLS FIRST
        metric_accessor = DBModel.metrics[sort_by].astext # ->>
        # Cast to Float for proper numerical sorting
        # Handle potential errors during cast if metric is not a valid number
        # We might need to handle missing keys more robustly
        query = query.order_by(
            order_func(sqlalchemy.cast(metric_accessor, Float)).nullslast() # NULLs в конце при сортировке
        )
    else:
        # Если sort_by не распознан, сортируем по умолчанию (start_time desc)
         logger.warning(f"Unsupported sort_by field: '{sort_by}'. Defaulting to start_time desc.")
         query = query.order_by(sqlalchemy.desc(DBModel.start_time))


    # Выполнение запроса
    try:
        db_models = query.all()
        logger.debug(f"Found {len(db_models)} models matching criteria.")
        # Преобразование в Pydantic модель
        return [TrainedModel.from_orm(db_model) for db_model in db_models]
    except Exception as e:
         logger.error(f"Error during database query for trained models: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Error retrieving trained models.")


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


@app.get("/download_dataset/{filename}") # без изменений
async def download_dataset(filename: str, db: Session = Depends(get_db)):
     # ... (код без изменений) ...
    # Добавим проверку существования записи в БД
    dataset_db = db.query(Dataset).filter(Dataset.filename == filename).first()
    if not dataset_db:
         raise HTTPException(status_code=404, detail=f"Dataset record for '{filename}' not found in database.")

    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Dataset file missing on disk: {filepath}, but DB record exists.")
        raise HTTPException(status_code=404, detail=f"Dataset file '{filename}' not found on server.")

    try:
        with open(filepath, "rb") as f:
            content = f.read()

        # Пытаемся определить Content-Disposition filename более корректно
        # Это помогает с не-ASCII символами в имени файла
        from fastapi.responses import StreamingResponse
        import urllib.parse

        # Кодируем имя файла для заголовка
        encoded_filename = urllib.parse.quote(filename)

        # Возвращаем как StreamingResponse для потенциально больших файлов
        # Но для простоты можно оставить Response
        # return StreamingResponse(io.BytesIO(content), media_type="text/csv", headers={
        #     "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}" # Стандартный способ кодирования
        # })
        return Response(content=content, media_type="text/csv", headers={
             "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{encoded_filename}"
         })

    except Exception as e:
        logger.exception(f"Error reading or serving dataset file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error serving dataset file.")


@app.delete("/dataset/{dataset_id}") # без изменений
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    logger.info(f"Attempting to delete dataset with ID: {dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logger.warning(f"Dataset with ID {dataset_id} not found for deletion.")
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_filename_to_delete = dataset.filename
    image_filename_to_delete = dataset.image_filename
    dataset_name_for_log = dataset.name or dataset.filename # Для логов

    # 1. Найти и удалить связанные обученные модели (файлы и записи в БД)
    associated_models = db.query(DBModel).filter(DBModel.dataset_filename == dataset_filename_to_delete).all()
    if associated_models:
        logger.info(f"Found {len(associated_models)} models associated with dataset '{dataset_name_for_log}'. Deleting them.")
        for model in associated_models:
            model_path = os.path.join(MODELS_DIR, model.model_filename)
            try:
                if os.path.exists(model_path):
                     os.remove(model_path)
                     logger.debug(f"Deleted model file: {model_path}")
                else:
                     logger.warning(f"Model file not found, skipping deletion: {model_path}")
                # Остановить и удалить инференс процесс, если он запущен для этой модели
                if model.id in inference_processes:
                    try:
                        logger.info(f"Stopping inference process for associated model {model.id}")
                        inference_processes[model.id].stop()
                        del inference_processes[model.id]
                    except Exception as stop_err:
                        logger.error(f"Error stopping inference process for model {model.id} during dataset deletion: {stop_err}")

                db.delete(model) # Удаляем запись модели из БД
            except OSError as e:
                 logger.error(f"Error deleting model file {model_path}: {e}")
                 # Продолжаем удаление других моделей и датасета, но логируем ошибку
            except Exception as e_model:
                logger.error(f"Error deleting model record {model.id} from DB: {e_model}")
                # Продолжаем

    # 2. Удалить файл датасета
    filepath = os.path.join(UPLOADS_DIR, dataset_filename_to_delete)
    try:
        if os.path.exists(filepath):
             os.remove(filepath)
             logger.debug(f"Deleted dataset file: {filepath}")
        else:
             logger.warning(f"Dataset file not found, skipping deletion: {filepath}")
    except OSError as e:
        logger.error(f"Error deleting dataset file {filepath}: {e}")
        # Можно решить, стоит ли продолжать, если файл не удалился
        # raise HTTPException(status_code=500, detail=f"Error deleting dataset file: {e}") # Остановить процесс

    # 3. Удалить файл изображения (если есть)
    if image_filename_to_delete:
        image_path = os.path.join(UPLOADS_DIR, image_filename_to_delete)
        try:
            if os.path.exists(image_path):
                 os.remove(image_path)
                 logger.debug(f"Deleted image file: {image_path}")
            else:
                 logger.warning(f"Image file not found, skipping deletion: {image_path}")
        except OSError as e:
            logger.error(f"Error deleting image file {image_path}: {e}")
            # Продолжаем удаление записи из БД

    # 4. Удалить запись о датасете из БД
    try:
        db.delete(dataset)
        db.commit()
        logger.info(f"Successfully deleted dataset '{dataset_name_for_log}' (ID: {dataset_id}) and associated models.")
        return {"message": f"Dataset '{dataset_name_for_log}' and associated models deleted successfully."}
    except Exception as e_db:
         logger.error(f"Error deleting dataset record {dataset_id} from DB: {e_db}")
         db.rollback() # Откатываем транзакцию
         raise HTTPException(status_code=500, detail="Error deleting dataset record from database.")



@app.post("/start_inference/{model_id}") # без изменений
async def start_inference_endpoint(model_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    global inference_processes
    logger.info(f"Request to start inference for model: {model_id}")

    # Проверяем, не запущен ли уже процесс
    if model_id in inference_processes:
         process_wrapper = inference_processes[model_id]
         if process_wrapper.process and process_wrapper.process.is_alive():
              logger.info(f"Inference process for model {model_id} is already running.")
              # Добавим URL для JSON запросов к документации Swagger
              json_docs_url = f"{app.openapi_url}" if hasattr(app, 'openapi_url') else "/docs#/default/predict_endpoint_predict__model_id__post" # Примерный путь
              return {
                    "message": "Inference process already running.",
                    "status": "running",
                    "model_id": model_id,
                    # TODO: Сделать URL динамическим или из конфигурации
                    "predict_json_url": f"/predict/{model_id}", # Прямой URL для POST JSON
                    "docs_url": json_docs_url # URL на Swagger UI
              }
         else:
              # Процесс есть в словаре, но мертв - удалим его перед перезапуском
              logger.warning(f"Found dead inference process for model {model_id}. Cleaning up.")
              try:
                   process_wrapper.stop() # Попытка корректно завершить, если возможно
              except Exception:
                   pass # Игнорируем ошибки при остановке мертвого процесса
              del inference_processes[model_id]


    # Ищем модель в БД
    db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not db_model or not db_model.model_filename:
        logger.error(f"Model {model_id} not found in DB or model filename is missing.")
        raise HTTPException(
            status_code=404,
            detail="Model not found or model data is missing"
        )

    # Проверяем наличие файла модели
    model_path = os.path.join(MODELS_DIR, db_model.model_filename)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found on disk for model {model_id}: {model_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Model file '{db_model.model_filename}' not found on server."
        )

    # Запускаем процесс
    try:
        result_queue = Queue()
        input_queue = Queue() # Добавляем input_queue в конструктор (если класс InferenceProcess обновлен)
        stop_event = Event()  # Добавляем stop_event (если класс InferenceProcess обновлен)

        # Убедимся, что класс InferenceProcess принимает все необходимые аргументы
        # Если конструктор старый:
        # inference_process = InferenceProcess(model_id, db_model.model_filename, result_queue)
        # Если конструктор обновлен (как в примере выше):
        inference_process = InferenceProcess(
             model_id=model_id,
             model_filename=db_model.model_filename,
             result_queue=result_queue
             # input_queue=input_queue, # Передаем, если класс ожидает
             # stop_event=stop_event   # Передаем, если класс ожидает
         )
        inference_process.start() # Запускает процесс в фоновом режиме
        inference_processes[model_id] = inference_process
        logger.info(f"Successfully started inference process for model {model_id} (PID: {inference_process.process.pid if inference_process.process else 'N/A'}).")

    except Exception as e:
        logger.exception(f"Failed to start inference process for model {model_id}:")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start inference process: {str(e)}"
        )

    json_docs_url = f"{app.openapi_url}" if hasattr(app, 'openapi_url') else "/docs#/default/predict_endpoint_predict__model_id__post"
    return {
        "message": "Inference process started successfully.",
        "status": "running",
        "model_id": model_id,
        "predict_json_url": f"/predict/{model_id}",
        "docs_url": json_docs_url
    }


@app.post("/predict/{model_id}") # без изменений
async def predict_endpoint(model_id: str, data: List[Dict[str, Union[float, int, str]]], db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    global inference_processes
    logger.debug(f"Received prediction request for model {model_id} with {len(data)} data points.")

    if model_id not in inference_processes:
        logger.warning(f"Prediction request for non-existent or stopped inference process: {model_id}")
        raise HTTPException(status_code=404, detail="Inference process not running or not found. Please start it first.")

    process_wrapper = inference_processes[model_id]

    # Дополнительная проверка, что процесс жив
    if process_wrapper.process is None or not process_wrapper.process.is_alive():
         logger.error(f"Inference process for model {model_id} found in registry but is not alive.")
         # Удаляем мертвый процесс из словаря
         del inference_processes[model_id]
         raise HTTPException(status_code=404, detail="Inference process was found but is not running. Please restart it.")

    # Проверка входных данных (базовая)
    if not data:
         raise HTTPException(status_code=400, detail="Input data list cannot be empty.")
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
         raise HTTPException(status_code=400, detail="Input data must be a list of dictionaries.")

    # TODO: Более строгая валидация данных - сравнение ключей словарей с ожидаемыми фичами модели
    try:
        # Получаем ожидаемые фичи (можно кэшировать для производительности)
        expected_features = await get_model_features(model_id, db) # Используем существующий эндпоинт/логику
        expected_feature_set = set(expected_features)

        # Проверяем первую запись данных на наличие всех фичей
        if data:
             first_item_keys = set(data[0].keys())
             if not expected_feature_set.issubset(first_item_keys):
                  missing = expected_feature_set - first_item_keys
                  extra = first_item_keys - expected_feature_set
                  error_msg = "Input data keys mismatch."
                  if missing: error_msg += f" Missing: {missing}."
                  if extra: error_msg += f" Unexpected: {extra}."
                  logger.error(f"Feature mismatch for model {model_id}. Expected: {expected_features}, Got: {list(first_item_keys)}")
                  raise HTTPException(status_code=400, detail=error_msg)
             # Можно добавить проверку типов данных, если необходимо

    except HTTPException as e:
         # Если get_model_features не сработал, пробрасываем ошибку
         if e.status_code == 404: # Модель не найдена при получении фич
             raise HTTPException(status_code=404, detail="Model not found (cannot verify features).")
         elif e.status_code == 500: # Ошибка при получении фич
             raise HTTPException(status_code=500, detail="Could not verify input features against model.")
         else: # Другая ошибка валидации фич
             raise e
    except Exception as e_feat:
         logger.error(f"Unexpected error during feature validation for prediction: {e_feat}", exc_info=True)
         # Решаем, блокировать ли предсказание или продолжить с предупреждением
         raise HTTPException(status_code=500, detail="Error validating input features.")


    result_queue = process_wrapper.result_queue
    try:
        process_wrapper.predict(data) # Отправляем данные в очередь процесса
        # Ожидаем результат из очереди с таймаутом
        result = result_queue.get(timeout=20) # Увеличим таймаут

        if result.get("error"):
            logger.error(f"Error received from inference process {model_id}: {result['error']}")
            # Возможно, стоит остановить процесс, если он выдал ошибку?
            # process_wrapper.stop()
            # del inference_processes[model_id]
            raise HTTPException(status_code=500, detail=f"Prediction failed: {result['error']}")

        predictions = result.get("predictions")
        if predictions is None:
             logger.error(f"Inference process {model_id} returned None for predictions.")
             raise HTTPException(status_code=500, detail="Prediction process returned no result.")

        logger.debug(f"Successfully received {len(predictions)} predictions from model {model_id}.")
        return {"predictions": predictions}

    except multiprocessing.queues.Empty: # Correct exception for queue timeout
        logger.error(f"Prediction request for model {model_id} timed out after 20 seconds.")
        # Процесс мог зависнуть. Возможно, стоит его остановить?
        # try:
        #     process_wrapper.stop()
        #     del inference_processes[model_id]
        # except Exception as stop_err:
        #      logger.error(f"Error stopping potentially hung process {model_id} after timeout: {stop_err}")
        raise HTTPException(status_code=504, detail="Prediction timed out. The model might be too slow or the process hung.") # 504 Gateway Timeout
    except Exception as e:
        logger.exception(f"Unexpected error during prediction communication with process {model_id}:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during prediction: {e}")


@app.delete("/stop_inference/{model_id}") # без изменений
async def stop_inference_endpoint(model_id: str):
     # ... (код без изменений) ...
    global inference_processes
    logger.info(f"Request to stop inference for model: {model_id}")

    if model_id not in inference_processes:
        logger.warning(f"Stop request for non-existent inference process: {model_id}")
        raise HTTPException(status_code=404, detail="Inference process not found.")

    process_wrapper = inference_processes[model_id]

    if process_wrapper.process is None or not process_wrapper.process.is_alive():
        logger.warning(f"Stop request for inference process {model_id}, but it's not running or already stopped.")
        # Удаляем запись из словаря, если процесс мертв
        del inference_processes[model_id]
        # Можно вернуть 200 OK, т.к. цель достигнута - процесс не работает
        return {"message": "Inference process was already stopped or not running."}
        # Или 404, если считать, что "активного" процесса нет
        # raise HTTPException(status_code=404, detail="Inference process is not currently running.")

    try:
        process_wrapper.stop() # Вызываем метод stop нашего класса
        # Дождемся завершения процесса (с таймаутом)
        if process_wrapper.process:
             process_wrapper.process.join(timeout=5)
             if process_wrapper.process.is_alive():
                  logger.warning(f"Inference process {model_id} did not terminate gracefully after stop signal. Forcing termination.")
                  process_wrapper.process.terminate() # Принудительное завершение
                  process_wrapper.process.join(timeout=2) # Ждем еще немного

        # Очищаем очереди и другие ресурсы, если это не делается в stop()
        # process_wrapper.input_queue.close()
        # process_wrapper.result_queue.close()

        # Удаляем процесс из словаря ПОСЛЕ успешной остановки
        del inference_processes[model_id]
        logger.info(f"Successfully stopped inference process for model {model_id}.")
        return {"message": "Inference process stopped successfully."}

    except Exception as e:
        logger.exception(f"Error occurred while trying to stop inference process {model_id}:")
        # Даже если была ошибка, пытаемся удалить из словаря, чтобы не было "зомби"
        if model_id in inference_processes:
             del inference_processes[model_id]
        raise HTTPException(status_code=500, detail=f"Failed to stop inference process: {str(e)}")


@app.get("/inference_status/{model_id}") # без изменений
async def inference_status_endpoint(model_id: str):
    # ... (код без изменений) ...
    global inference_processes
    if model_id not in inference_processes:
        return {"status": "not_found"} # Явный статус "не найден"

    process_wrapper = inference_processes[model_id]

    if process_wrapper.process and process_wrapper.process.is_alive():
        # Дополнительно можно проверить, отвечает ли процесс (сложно)
        # Или вернуть PID
        return {"status": "running", "pid": process_wrapper.process.pid}
    else:
        # Процесс есть в словаре, но мертв - не консистентное состояние
        logger.warning(f"Inference status check found dead process for model {model_id}.")
        # Можно удалить его здесь
        # del inference_processes[model_id]
        return {"status": "stopped_unexpectedly"} # Или "error", "stopped"


@app.get("/running_models/", response_model=List[RunningModel]) # без изменений
async def get_running_models(db: Session = Depends(get_db)): # Добавим DB зависимость
    """Returns a list of currently running inference models with details from DB."""
    # ... (код без изменений) ...
    running_models_list = []
    # Создаем копию ключей, чтобы избежать проблем при удалении во время итерации
    current_process_ids = list(inference_processes.keys())

    for model_id in current_process_ids:
        if model_id not in inference_processes: # Проверка, если процесс был удален в другом запросе
             continue

        inference_process = inference_processes[model_id]

        if inference_process.process and inference_process.process.is_alive():
            # Запрашиваем информацию о модели из БД
            db_model = db.query(DBModel).filter(DBModel.id == model_id).first()
            if db_model:
                try:
                    # Формируем RunningModel с использованием .from_orm или вручную
                    # Убедимся, что все поля есть в DBModel или могут быть выведены
                    running_model_data = RunningModel(
                        model_id=model_id,
                        dataset_filename=db_model.dataset_filename,
                        target_column=db_model.target_column,
                        model_type=db_model.model_type,
                        metrics=db_model.metrics if db_model.metrics else {}, # Обработка None
                        status="running",
                        api_url=f"/inference/{model_id}" # Используем относительный URL
                    )
                    running_models_list.append(running_model_data)
                except ValidationError as e:
                     logger.error(f"Error creating RunningModel Pydantic object for model {model_id}: {e}")
                     # Пропустить эту модель или добавить с неполными данными? Пропускаем.
                except Exception as e_orm:
                     logger.error(f"Error accessing attributes for DBModel {model_id}: {e_orm}")

            else:
                logger.warning(f"Running inference process found for model ID {model_id}, but no corresponding record in DBModel table.")
                # Можно добавить запись с неполной информацией или пропустить
                # running_models_list.append(RunningModel(model_id=model_id, status="running_orphan", ... ))
        else:
            # Если процесс мертв, но все еще в словаре, удаляем его
            logger.warning(f"Cleaning up dead inference process entry for model ID {model_id}.")
            del inference_processes[model_id]


    return running_models_list


@app.post("/datasets/{dataset_id}/visualize", status_code=202) # без изменений
async def start_visualization_generation(dataset_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    logger.info(f"Received request to start visualization for dataset_id: {dataset_id}")
    # Проверка, существует ли датасет
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logger.warning(f"Dataset {dataset_id} not found for visualization request.")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Проверим статус существующей визуализации, если есть
    existing_viz = db.query(DatasetVisualization)\
                     .filter(DatasetVisualization.dataset_id == dataset_id)\
                     .order_by(DatasetVisualization.generated_at.desc())\
                     .first()

    # Не перезапускаем, если уже PENDING или RECENTLY SUCCESS? (опционально)
    if existing_viz and existing_viz.status == "PENDING":
         # Можно проверить ID задачи Celery, если сохраняли его
         logger.info(f"Visualization task for dataset {dataset_id} is already pending.")
         # Возвращаем 200 OK или 202 Accepted? 200 с сообщением, что уже запущено.
         # Чтобы вернуть task_id, нужно его где-то хранить... Пока просто 200.
         return Response(status_code=200, content=json.dumps({"message": "Visualization task is already pending."}), media_type="application/json")
         # raise HTTPException(status_code=409, detail="Visualization task is already pending.") # 409 Conflict

    # Если статус FAILURE или SUCCESS (или нет записи), запускаем новую задачу
    logger.info(f"Queueing visualization task for dataset_id: {dataset_id}")
    try:
        task = generate_visualizations_task.delay(dataset_id)
        logger.info(f"Visualization task {task.id} queued successfully for dataset_id: {dataset_id}")

        # Можно обновить/создать запись в DatasetVisualization со статусом PENDING и task_id
        if existing_viz:
            existing_viz.status = "PENDING"
            existing_viz.error_message = None
            existing_viz.visualization_data = None
            # existing_viz.celery_task_id = task.id # Если добавили поле
            db.commit()
        else:
            # Создаем новую запись сразу
            new_viz_entry = DatasetVisualization(
                 dataset_id=dataset_id,
                 status="PENDING"
                 # celery_task_id=task.id # Если добавили поле
            )
            db.add(new_viz_entry)
            db.commit()


        return {"task_id": task.id, "message": "Visualization task successfully queued."}
    except Exception as e:
         logger.error(f"Failed to queue Celery task for visualization: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to start visualization task.")


@app.get("/datasets/{dataset_id}/visualizations", response_model=Optional[VisualizationDataResponse]) # без изменений
async def get_visualizations(dataset_id: str, db: Session = Depends(get_db)):
    # ... (код без изменений) ...
    logger.debug(f"Fetching visualization data for dataset_id: {dataset_id}")
    visualization = db.query(DatasetVisualization)\
                      .filter(DatasetVisualization.dataset_id == dataset_id)\
                      .order_by(DatasetVisualization.generated_at.desc())\
                      .first()

    if not visualization:
        logger.debug(f"No visualization record found for dataset_id: {dataset_id}")
        # Важно вернуть null или объект со статусом "NOT_STARTED"?
        # Возвращаем null, фронтенд поймет, что надо запускать.
        return None

    logger.debug(f"Found visualization record for dataset_id {dataset_id} with status: {visualization.status}")
    # Используем Pydantic модель для формирования ответа
    try:
        return VisualizationDataResponse.from_orm(visualization)
    except ValidationError as e:
         logger.error(f"Error converting visualization ORM object to Pydantic model for dataset {dataset_id}: {e}")
         # Возвращаем ошибку или данные как есть? Лучше ошибку.
         raise HTTPException(status_code=500, detail="Error retrieving visualization data format.")


@app.get("/visualization_status/{task_id}") # без изменений
async def get_visualization_status(task_id: str):
     # ... (код без изменений) ...
    logger.debug(f"Checking status for visualization task_id: {task_id}")
    task_result = AsyncResult(task_id, app=celery)

    status = task_result.status
    result = task_result.result
    error_info = None

    if task_result.failed():
         status = "FAILURE"
         if isinstance(result, dict) and 'exc_type' in result and 'exc_message' in result:
            error_info = {
                 'type': result.get('exc_type', 'UnknownError'),
                 'message': result.get('exc_message', 'No message'),
                 'traceback': result.get('exc_traceback', None) # Исправлено имя ключа
             }
         elif isinstance(result, Exception):
             error_info = {
                 'type': type(result).__name__,
                 'message': str(result),
                 'traceback': getattr(task_result, 'traceback', None)
             }
             if not error_info['traceback'] and hasattr(task_result.backend, 'get_task_meta'):
                  try:
                      meta = task_result.backend.get_task_meta(task_id)
                      error_info['traceback'] = meta.get('traceback')
                  except Exception: pass
         else:
             error_info = {
                 'type': "UnknownError",
                 'message': str(result) if result else "Unknown failure reason",
                 'traceback': getattr(task_result, 'traceback', None)
             }
         result = None # Очищаем результат при ошибке

    elif task_result.successful():
        status = "SUCCESS"
        # Проверяем, вернула ли таска словарь с visualization_db_id
        if not isinstance(result, dict) or ('visualization_db_id' not in result and 'message' not in result) :
             logger.warning(f"Visualization task {task_id} succeeded but returned unexpected result format: {result}")
             # Можно вернуть статус SUCCESS, но с предупреждением в result
             # result = {"warning": "Task succeeded but result format is unexpected.", "original_result": result}


    response = {
        "task_id": task_id,
        "status": status,
        "result": result, # Результат только при успехе (или с предупреждением)
        "error": error_info
    }
    logger.debug(f"Status for visualization task {task_id}: Status={response['status']}, Error={response['error'] is not None}")
    return response