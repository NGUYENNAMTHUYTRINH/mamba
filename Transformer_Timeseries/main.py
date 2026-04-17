# -*- coding: utf-8 -*-
# ---------------------

import click
import torch.backends.cudnn as cudnn

from conf import Conf
from trainer import Trainer
from inference import TS

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default='conf/air_quality.yaml')
@click.option('--seed', type=int, default=None)
@click.option('--inference', type=bool, default=False)
@click.option('--location', type=str, default=None, help='Nhập mã/tên địa điểm bạn muốn train. Ví dụ: HaNoi') # THÊM OPTION NÀY
def main(exp_name, conf_file_path, seed, inference, location):
    # type: (str, str, int, bool, str) -> None

    # if `exp_name` is None, ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('experiment name', default='default')

    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)
    
    # Gán biến location vừa nhập từ dòng lệnh vào cấu hình cnf
    cnf.selected_location = location

    print(f'\n{cnf}')
    print(f"\nStarting Experiment '{exp_name}' [seed: {cnf.seed}]")
    if location:
        print(f'Selected Location: {location}')

    if inference:
        ts_model = TS(cnf=cnf)
        ts_model.test()
    else:
        trainer = Trainer(cnf=cnf)
        trainer.run()


if __name__ == '__main__':
    main()