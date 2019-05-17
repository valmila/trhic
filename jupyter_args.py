
class args:
    batch_size=64
    dataset_size=60000
    dataset='mnist'
    estim_size=5000
    model='cnn'
    lr=0.0002
    momentum=0.9
    mode='inv'
    rcond=1e-8
    corrupt=0e-8
    reg=1e-2
    cuda = torch.cuda.is_available()
