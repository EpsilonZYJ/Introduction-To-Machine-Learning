from Utils import *

def main(is_debug=True):
    set_debug_mode(True)
    train_feature = load_data('data.csv')
    train_label = load_data('targets.csv')
    print(train_feature.shape)
    print(train_label.shape)
    print(train_feature)
    print(train_label)

if __name__ == '__main__':
    main(True)
