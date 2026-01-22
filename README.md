# Digit Recognition Neural Network

A neural network that recognizes handwritten digits! This project trains a model on the **MNIST dataset** using PyTorch and provides a **front-end interface** where users can draw a number and get a prediction.

![MNIST Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)  
*Example digits from the MNIST dataset.*

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- Trains a convolutional neural network on the MNIST dataset.
- Front-end interface to draw digits and see predictions in real-time.
- Built with **PyTorch**, **FastAPI**, and simple **HTML/CSS** frontend.

---

## Installation

### Prerequisites
Make sure you have **Python 3.8+** installed. Then install the required packages:

```bash
pip install torch torchvision numpy fastapi uvicorn
```

---

## Usage

Basic usage examples:

- Train the model:
```bash
python train.py
```

- Run the FastAPI backend:
```bash
uvicorn main:app --reload
```

- Open the frontend (for example, open frontend/index.html in a browser or visit the served frontend if configured).

Example training loop snippet from train.py:
```python
def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = lossFunction(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_index % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders['train'].dataset)}"
                  f" ({100. * batch_index / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")
```

---

## Project Structure

```
Digit-Recognition-NN/
│
├─ data/                # MNIST dataset (downloaded automatically)
├─ models/              # Saved neural network models
├─ frontend/            # HTML/CSS files for drawing interface
├─ main.py              # FastAPI backend
├─ train.py             # Training script
└─ README.md
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m "Add feature"`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a Pull Request

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
