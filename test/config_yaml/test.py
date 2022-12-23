from omegaconf import OmegaConf

config = OmegaConf.load(f'configs/baseline.yaml')

assert(config.model.metric_list.metric1 == 'klue_re_micro_f1')

print('Test passed.')
