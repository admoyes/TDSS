from torchvision import transforms


def DefaultTransform():

    return transforms.Compose([
        transforms.ToTensor()
    ])