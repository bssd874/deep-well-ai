import time
import io
import json
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChatResponse
from app.services.inference import DeepWellInference

app = FastAPI(title="DEEP-WELL AI API", version="3.1.0")
engine = DeepWellInference()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def prepare_dataframe(raw_data: list) -> pd.DataFrame:
    # Directly create DataFrame from list of dicts
    df = pd.DataFrame(raw_data)
    # Minimal cleaning: drop fully empty rows
    df.dropna(how='all', inplace=True)
    return df

@app.post("/api/v1/chat", response_model=ChatResponse)
async def unified_chat(
    model: str = Form(...),
    messages: Optional[str] = Form(None),
    files: List[UploadFile] = File(None) 
):
    if not engine.is_valid_model(model):
        raise HTTPException(404, f"Model '{model}' not found.")

    aggregated_data = []
    if messages:
        try:
            clean_messages = messages.replace('\\n', '\n').strip()
            try:
                data = json.loads(clean_messages)
                if isinstance(data, list): aggregated_data.extend(data)
                elif isinstance(data, dict):
                    if "log_data" in data: aggregated_data.extend(data["log_data"])
                    else: aggregated_data.append(data)
            except json.JSONDecodeError:
                # Parse raw text to DataFrame first, then to dict for uniformity
                # Ideally we keep it as DF, but for simplicity of merging with files, we use list of dicts
                df = pd.read_csv(io.StringIO(clean_messages), sep=r'\s+|,|\t', engine='python')
                aggregated_data.extend(df.to_dict(orient='records'))
        except Exception as e:
            raise HTTPException(400, f"Text parsing failed: {str(e)}")

    if files:
        for file in files:
            try:
                content = await file.read()
                if file.filename.endswith('.xlsx'): df = pd.read_excel(io.BytesIO(content))
                elif file.filename.endswith('.csv'): df = pd.read_csv(io.BytesIO(content))
                else: continue
                aggregated_data.extend(df.to_dict(orient='records'))
            except Exception as e:
                raise HTTPException(400, f"Error in file '{file.filename}': {str(e)}")
    if not files:
        files = []

    if not aggregated_data:
        raise HTTPException(400, "No valid data found in any input source.")

    # Create ONE DataFrame for the entire batch
    batch_df = prepare_dataframe(aggregated_data)
    
    # Run prediction on DataFrame
    prediction_result = await engine.predict_unified(batch_df, model)

    return {
        "id": f"dw-{int(time.time())}",
        "created": int(time.time()),
        "model": model,
        "choices": [{"message": prediction_result, "finish_reason": "stop"}],
        "usage": {
            "total_records": len(batch_df), 
            "input_type": prediction_result.get("detection_type"),
            "status": "processed",
            "warnings": prediction_result.get("warnings", [])
        }
    }