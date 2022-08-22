import argparse
import torch
import embed


def main(args):
    if args.gpu:
        torch.cuda.set_device(0)
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    
    with open(args.data_path, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    
    # emb is with the shape of (len(data), 1024)
    emb = embed.encode(data, args.model_path, args.model_name, args.language, device)
    print(emb.shape)

    """
    Your code here.
    XXX XXX XXX
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='language', default='en', help='language', required=True, type=str)
    parser.add_argument('-m', dest='model_name', default='EMS', help='model name', type=str)
    parser.add_argument('--model_path', dest='model_path', default='../ckpt/EMS_model.pt', help='path of the model', type=str)
    parser.add_argument('--data_path', dest='data_path', default=None, help='path of the data', required=True, type=str)
    parser.add_argument('--gpu', dest='gpu', help='use gpu or not', action='store_true')
    args = parser.parse_args()
    main(args)

