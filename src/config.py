#####config training parameters
chexper_params= {'batch_size': 16,
                'lr': 1e-5,
                'epoches': 12,
                'image_size': 320}

label_strategy = ['u-Zeros','u-Ones','u-Ignore','u-MultiClass','u-SelfTrained']

input_file = {'train': '../CheXpert-v1.0-small/train.csv',
            'test': '../CheXpert-v1.0-small/valid.csv',
             'infer':'../CheXpert-v1.0-small/infer.csv' }

output_path = {'u-Zeros':{'directory':'uZeros',
                        'checkpoint_path':'checkpoint',
                        'model_name':'uZeros.h5'},
                'u-Ones':{'directory':'uOnes',
                        'checkpoint_path':'checkpoint',
                        'model_name':'uOnes.h5'},
                'u-Ignore':{'directory':'uIgnore',
                        'checkpoint_path':'checkpoint',
                        'model_name':'uIgnore.h5'},
                'u-MultiClass':{'directory':'uMultiClass',
                            'checkpoint_path':'checkpoint',
                            'model_name':'uMultiClass.h5'},
                'u-SelfTrained':{'directory':'uSelfTrained',
                                'checkpoint_path':'checkpoint',
                                'model_name':'uSelfTrained.h5'}}

train_approach = {'finetune':False,
                    'chexnet': True,
                    'efficientnet':True}
