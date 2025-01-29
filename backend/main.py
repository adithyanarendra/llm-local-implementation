from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 150


@app.get("/")
async def root():
    return {"message": "Welcome to the Local LLM API"}


@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt",
                           padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=request.max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
