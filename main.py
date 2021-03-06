import torchvision.models as models
from models.ResFormerNet import resformer, train_model


if __name__ == '__main__':

    train_model(5, resformer)