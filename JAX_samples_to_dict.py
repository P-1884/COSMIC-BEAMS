def JAX_samples_to_dict(sampler,separate_keys=False,cosmo_type=''):
    key_list = sampler.get_samples().keys()
    sample_dict = {}
    for k_i in key_list:
        if 'unscaled' in k_i: continue
        if not separate_keys: 
            assert sampler.get_samples()[k_i].shape[1]==1 and len(sampler.get_samples()[k_i].shape)==2
            sample_dict[k_i] = sampler.get_samples()[k_i].T[0]
        else: 
            print(k_i,sampler.get_samples(True)[k_i].shape)
            if k_i not in ['Ok','zL','zS']: 
                if k_i=='Ok': assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==3
                if k_i in ['zL','zS']: assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==4
            for c_i in range(sampler.get_samples(True)[k_i].shape[0]):
                try:
                    if k_i not in ['zL','zS']:
                        sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0]
                        print('Saved shape 1:',k_i,c_i,sampler.get_samples(True)[k_i][c_i,:,0].shape)
                    #May require this if using photometric redshifts
                    if k_i in ['zL','zS']:
                        #print('Shape 2.0:',sampler.get_samples(True)[k_i].shape) #(2, 100, 1, 2048)
                        for z_i in range(sampler.get_samples(True)[k_i].shape[-1]):
                            if z_i>=100: 
                                print(f'Skipping {k_i}_{z_i}_{c_i}')
                                continue #Only saving the first 100 redshifts
                            sample_dict[f'{k_i}_{z_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0,z_i]
                            print('Saved shape 2:',f'{k_i}_{z_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:,0,z_i].shape)
                except:
                    #May require this exception if using FlatwCDM or FlatLambdaCDM
                    print('Exception here',k_i,cosmo_type)
                    assert (k_i=='Ok') or (k_i in ['w','wa'] and cosmo_type in ['LambdaCDM','FlatLambdaCDM'])
                    sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:]
                    print('Saved shape 3:',f'{k_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:].shape)
    return sample_dict