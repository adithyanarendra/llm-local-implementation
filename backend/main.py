from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

app = FastAPI()


class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50


@app.get("/")
async def root():
    return {"message": "Welcome to the Local LLM API"}


@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=request.max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
