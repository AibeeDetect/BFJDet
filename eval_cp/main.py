import argparse
from eval_mmr import eval_mmr
from eval_mr_cityperson import eval_mr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt_path', '-dt', default="")
    parser.add_argument('--gt_path', '-gt', default="")
    parser.add_argument('--type', '-t', default='mmr')

    args = parser.parse_args()
    if args.type == 'mr_body':
        eval_mr(args.gt_path, args.dt_path, 'body')
    elif args.type == 'mr_face':
        eval_mr(args.gt_path, args.dt_path, 'face')
    elif args.type == 'mmr':
        eval_mmr(args.gt_path, args.dt_path)


if __name__ == '__main__':
    main()
