from .model.resnet1D import ResNet1D

def main():
    model = ResNet1D(1, 2)

    my_dict = model.state_dict()

    for k, v in my_dict.items():
        print(k, v.shape)

if "__name__" == "__main__":
    main()