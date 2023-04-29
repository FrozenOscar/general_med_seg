import os
import re
from torchvision import transforms


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def check_path(*path_list):
    for path in path_list:
        assert os.path.exists(path), f"path '{path}' does not exist"


def sort(v_list):
    def str2int(v_str):
        def tryint(s):
            try:
                return int(s)
            except ValueError:
                return s
        return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
    return sorted(v_list, key=str2int)


def print_log(args):
    print(f'.  Using amp.') if args.amp else print()
    if args.save:
        print(f'save model: {args.save}, save every {args.save_step} times')
    else:
        print(f'save model: {args.save}')
    print('model arguments:')
    print(f'backbone: {args.backbone}')
    print('training arguments:')
    print(f'batch_size: {args.batch_size}   lr: {args.lr}    lr_scheduler: {args.lr_scheduler}')


def read_log(log_path: str):
    with open(log_path, 'r') as f:
        logs = []
        for line in f.readlines():
            line = line.rstrip('\n')
            logs.append(line)
        return logs


def write_log(log_list: list, log_path: str, describe=''):
    with open(log_path, 'w') as f:
        for i, item in enumerate(log_list):
            log = f'{describe}{item}\n'
            f.write(log)


def Append_log(append_log, log_path: str, describe=''):
    with open(log_path, 'a') as f:
        if isinstance(append_log, str):
            log = f'{describe}{append_log}\n'
            f.write(log)

        if isinstance(append_log, list):
            for i, item, in enumerate(append_log):
                log = f'{describe}{item}\n'
                f.write(log)


def tensor_to_ndarray(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.data.float().cpu().clone()
    images = []
    if image.ndim == 4:
        for i in range(len(image)):
            img = image[i]
            img = unloader(img)
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[None, ...]        # 加上通道维度
            images.append(img)
    return images

if __name__ == '__main__':
    pass
