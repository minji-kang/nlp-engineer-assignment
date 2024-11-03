from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

import torch
from .transformer import CustomTransformer
from .dataset import CharCountDataset

app = FastAPI(
    title="NLP Engineer Assignment",
    version="1.0.0"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CustomTransformer.from_pretrained(
    checkpoint_path='data/trained_model.ckpt',
    vocab_size=27,
    device=device,
  )
model.eval()

@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")

@app.post('/character/predict')
async def predict(text: str):
    dataset = CharCountDataset(text)
    # Tokenize text into character indices
    input_seq = [dataset.char_to_index(char) for char in text]
    # Convert lists to tensors
    input_tensor = torch.tensor(input_seq, dtype=torch.long).to(device)
    
    logits, prediction = model.generate(input_tensor)
    
    return {
        'input': text,
        'prediction': ", ".join(map(str, prediction[0].tolist()))
    }

nest_asyncio.apply()