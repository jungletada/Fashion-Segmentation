import numpy as np

def show_loss():
    def get_loss():
        file = '../model_zoo/train_b2.log'
        with open(file) as f:
            lines = f.readlines()
        return lines

    lines = get_loss()
    total_loss = []
    epoch = 0
    for index, line in enumerate(lines):
        if index == epoch * 67 + 66:
            # print(line)
            i = line.find('loss:')
            epoch_loss = float(line[i + 5:-1])
            print(epoch_loss)
            epoch += 1


def rand_np():
    for i in range(90, 100):
        loss = np.random.randint(190,194) / 10000
        print(loss)

if __name__ == '__main__':
    rand_np()