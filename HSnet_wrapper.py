
import train as train
import test as test
import sys

if __name__ == "__main__":
    if sys.argv[0] == "test":
        test.main(sys.argv[1:])
    elif sys.argv[0] == "train":
        train.main(sys.argv[1:])
    else:
        print("only 'test' and 'train' are supported commands")