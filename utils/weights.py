import torch

if __name__ == '__main__':
    w = [0.009564808334435267,
         0.8668511413669072,
         8.084233179935099,
         2.504232633306998,
         1.1814753228475352,
         0.2536907054637364,
         0.3041276481502543,
         16.26010594399163,
         0.32602623753226145,
         0.2759930929558367,
         2.312275154503584,
         0.5081653592241218,
         6.76902925342786,
         3.0421933623991984]
    class_weights = torch.FloatTensor(w)
    path = './ModaNet_class_weights.pth'
    torch.save(class_weights, path)