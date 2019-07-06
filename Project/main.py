import numpy as np
import keras

from data import dataset


def main():
    d = dataset()
    print(list(d))


if __name__ == "__main__":
    main()

