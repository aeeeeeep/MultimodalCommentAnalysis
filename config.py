class Config(dict):
    def version_config(self, version):
        batch = 16
        val_batch = 8
        hp = {1: {'n_epoch':8, 'batch': batch, 'valid_batch': val_batch, 'n_layer':6},
              }
        self['n_epoch'] = hp[version].get('n_epoch', 8)
        self['n_layer'] = hp[version].get('n_layer', 6)
        self['batch'] = hp[version].get('batch', batch)
        self['valid_batch'] = hp[version].get('valid_batch', val_batch)
        self['w_g'] = 1

        #请自己造训练测试集
        self['train_file'] = './data/train.csv'
        self['valid_file'] = './data/val.csv'
    
        self['input_l'] = 512
        self['n_token'] = 50265
        self['sos_id'] = 0
        self['eos_id'] = 2
        self['pad_id'] = 1
        
    def __init__(self, version, seed=0):
        self['lr'] = 1e-4
        self['model_dir'] = './checkpoint/%d'%version
        if seed>0:
            self['model_dir'] += '_%d'%seed
        self['output_dir'] = './outputs/%d'%version
        
        self.version_config(version)
