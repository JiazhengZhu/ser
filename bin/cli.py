from pathlib import Path
import torch
from torch import optim

from ser.model import Net
from ser.transform import normalise
from ser.data import get_training_dataloader, get_validation_dataloader
from ser.train import begin_training

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        10, "-t", "--epochs", help="Number of time steps to train."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for training."
    ),
    learning_rate: float = typer.Option(
        0.01, "-r", "--learning-rate", help="Learning rate for training." 
    )
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment {name} on {device} device...")
    print(f"Config: epochs = {epochs}, batch_size = {batch_size}, learning rate = {learning_rate}. ")
    # epochs = 2
    # batch_size = 1000
    # learning_rate = 0.01

    # save the parameters!

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = normalise()

    # dataloaders
    training_dataloader = get_training_dataloader(ts, batch_size)
    validation_dataloader = get_validation_dataloader(ts, batch_size)

    # train
    begin_training(model, training_dataloader, validation_dataloader, epochs, optimizer, device)



@main.command()
def infer():
    print("This is where the inference code will go")
