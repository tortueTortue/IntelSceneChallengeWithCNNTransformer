"""
Here are the training methods.
"""
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD

from utils.utils import batches_to_device, get_default_device, to_device, save_checkpoints
from metrics.metrics import accuracy

# TODO : Add configs for this one
PATH = ""

def validation_step(model, batch):
    images, labels = batch 
    images, labels = images.cuda(), labels.cuda()
    out = model.forward(images)
    cross_entropy = CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: Module, val_set: DataLoader):
    outputs = [validation_step(model, batch) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def train(epochs_no: int, model: Module, train_set: DataLoader, val_set: DataLoader):
    history = []
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs_no):

        """  Training Phase """ 
        for batch in train_set:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            curr_loss = loss(model.forward(inputs), labels)
            curr_loss.backward()
            optimizer.step()

        """ Validation Phase """
        result = evaluate(model, val_set)
        print(result)
        history.append(result)
        if epoch % 10 == 0 :
            save_checkpoints(epoch, model, optimizer, loss, PATH)
    
    return history

def train_model(epochs_no: int, model_to_train: Module, model_name: str, dataset: Dataset, batch_size: int):
    model_to_train.train()
    device = get_default_device()

    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size)

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(model_to_train, device)

    train(epochs_no, model, train_loader, val_loader)

    return model