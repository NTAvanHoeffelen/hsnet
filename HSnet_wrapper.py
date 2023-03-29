
import train as train
import test as test
import sys

if __name__ == "__main__":
    if sys.argv[1] == "test":
        test.main(sys.argv[2:])
    elif sys.argv[1] == "train":
        train.main(sys.argv[2:])
    else:
        print("only 'test' and 'train' are supported commands")