from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim, sign


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        lr = 0.01
        for epoch in range(1000):
            converge = True
            for sample in dataloader:
                pred = sign(model(sample['x'])) # sign must be used to avoid precision issues
                if pred == sample["label"]:
                    pass
                else:
                    converge = False
                    model.w += lr * sample["label"] * sample["x"]

            if converge: return


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    lr = 1e-4
    epoch = 1000

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch in dataloader:
            optimizer.zero_grad()

            y_pred = model(batch["x"])
            loss = regression_loss(batch["label"], y_pred)
        
            loss.backward()

            optimizer.step()


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    lr = 2e-3
    epoch = 10

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch in dataloader:
            optimizer.zero_grad()

            y_pred = model(batch["x"])
            loss = digitclassifier_loss(batch["label"], y_pred)
        
            loss.backward()

            optimizer.step()

            if dataset.get_validation_accuracy() > 0.98:
                return

def train_languageid(model, dataset):
    print("Starting training...")
    model.train()
    lr = 0.001
    epoch = 20
    
    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting {epoch} epochs with {len(dataloader)} batches per epoch")

    for e in range(epoch):
        print(f"\n=== Epoch {e+1}/{epoch} ===")
        batch_count = 0
        
        for batch in dataloader:
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}...", end='', flush=True)
            
            optimizer.zero_grad()
            
            x, y = batch["x"], batch["label"]
            x = movedim(x, 0, 1)
            
            y_pred = model(x)
            loss = languageid_loss(y_pred, y)
        
            loss.backward()
            optimizer.step()
            
            if batch_count % 10 == 0:
                print(f" loss={loss.item():.4f}")
            
            batch_count += 1

            validation_accuracy = dataset.get_validation_accuracy()
            print(f"  Validation accuracy: {validation_accuracy:.4f}")
            if validation_accuracy >= 0.83:
                return

        
        print(f"Epoch {e+1} complete: {batch_count} batches processed")
    
    print("Training complete!")

def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    model.train()
    lr = 0.005
    epoch = 10

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch in dataloader:
            optimizer.zero_grad()

            y_pred = model(batch["x"])
            loss = digitconvolution_Loss(y_pred, batch["label"])
        
            loss.backward()

            optimizer.step()

            acc = dataset.get_validation_accuracy()

            print(f"Validation accuracy: {acc}")
            if acc > .80:
                return
