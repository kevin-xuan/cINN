from data import dataloader_RLBench, dataloader_bair, dataloader_bair_endpoint, dataloader_iPER, dataloader_landscape, dataloader_DTDB

def get_loader(name, control=False):

    if name == 'BAIR' or name == 'bair':
        return dataloader_bair_endpoint if control else dataloader_bair
    elif name == 'iper' or name == 'iPER':
        return dataloader_iPER
    elif name == 'landscape' or name == 'Landscape':
        return dataloader_landscape
    elif name == 'DTDB' or name == 'dtdb':
        return dataloader_DTDB
    elif name == 'RLBench':
        return dataloader_RLBench
    else:
        raise NotImplementedError(f'Corresponding dataloader to dataset {name} not implemented')


def get_eval_loader(name, length, path, config, control=False):

    config.Data['sequence_length'] = length
    config.Data['data_path'] = path

    if name == 'BAIR' or name == 'bair':
        return dataloader_bair_endpoint.Dataset(config, mode='test') if control else dataloader_bair.Dataset(config, mode='test')
    elif name == 'iper' or name == 'iPER':
        return dataloader_iPER.DatasetEvaluation(seq_length=length, img_size=config.Data['img_size'], path=path)
    elif name == 'landscape' or name == 'Landscape':
        return dataloader_landscape.Dataset(config, mode='test')
    elif name == 'DTDB' or name == 'dtdb':
        return dataloader_DTDB.Dataset(config, mode='test')
    elif name == 'RLBench' or name == 'rlbench':
        return dataloader_RLBench.Dataset(config, mode='test')
        # return dataloader_RLBench.Dataset(config, mode='train')
    else:
        raise NotImplementedError(f'Corresponding dataloader to dataset {name} not implemented')