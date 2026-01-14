from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where server.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Serve the frontend
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(SCRIPT_DIR, "index.html"))

# Define the same CNN architecture as in training
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=2)   

        self.fc1 = nn.Linear(1920, 84)                    
        self.fc2 = nn.Linear(84, 10)                     

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = torch.tanh(self.conv3(x))

        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)

# Load trained weights
weights_path = os.path.join(SCRIPT_DIR, "mnist_cnn_weights.pth")
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

class PredictionRequest(BaseModel):
    pixels: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert pixel array to tensor
    pixels = np.array(request.pixels, dtype=np.float32)
    
    # Reshape to 28x28
    pixels = pixels.reshape(1, 1, 28, 28)
    
    # Convert to tensor
    tensor = torch.from_numpy(pixels).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.exp(output)  # Convert log_softmax to probabilities
        confidence, prediction = torch.max(probabilities, 1)
    
    return PredictionResponse(
        prediction=prediction.item(),
        confidence=confidence.item()
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
