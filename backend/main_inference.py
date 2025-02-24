# backend/main_inference.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io
import joblib
import os

app = FastAPI()

MODEL_PATH = "model.joblib"  # Путь к модели внутри контейнера
model = None

# Загрузка модели при старте приложения
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}") # Логирование
    # Можно завершить приложение, если модели нет, но в этом случае, нужно чтобы docker сам пересоздал его, указав в docker-compose.yml
    # raise  # Или обработать по-другому

def validate_input_data(df, expected_columns):
    """Проверяет, что входной DataFrame содержит все необходимые колонки."""
    if not set(expected_columns).issubset(df.columns):
        missing_cols = set(expected_columns) - set(df.columns)
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_cols)}")

@app.post("/predict_uploadfile/")
async def predict_uploadfile(file: UploadFile = File(...)):
    """
    Принимает CSV файл, делает предсказания и возвращает CSV файл с результатами.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Must be CSV.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        validate_input_data(df, model.feature_names_in_) # Проверка
        predictions = model.predict(df)
        df['predictions'] = predictions # Добавляем колонку
         # Возвращаем CSV файл
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)  # Переходим в начало
        return StreamingResponse(
            iter([output.getvalue()]),  #
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
        )

    except (ValueError, KeyError) as e: # Обработка ошибок
       raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(data: list):
    """Принимает JSON, делает предсказания и возвращает JSON с результатами."""
    if not model:
       raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame(data)
        validate_input_data(df, model.feature_names_in_)
        predictions = model.predict(df)
        df['predictions'] = predictions
        result_df = df[['predictions']] # предсказание
        return JSONResponse(content=result_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
