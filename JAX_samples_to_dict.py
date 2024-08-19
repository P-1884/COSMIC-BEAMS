import numpy as np
def JAX_samples_to_dict(sampler,separate_keys=False,cosmo_type='',wa_const=False,w0_const=False,fixed_GMM=False,N_sys_per_batch = []):
    key_list = sampler.get_samples().keys()
    sample_dict = {}
    for k_i in key_list:
        if 'unscaled' in k_i: continue
        if not separate_keys: 
            assert sampler.get_samples()[k_i].shape[1]==1 and len(sampler.get_samples()[k_i].shape)==2
            sample_dict[k_i] = sampler.get_samples()[k_i].T[0]
        if len(N_sys_per_batch)>0:
            if ('zL' in k_i) or ('zS' in k_i):
                batch_n = int(k_i.split('_B')[-1])
                if batch_n>0: continue #Don't save any of the later batches
                assert k_i == f'{k_i.split("_B")[0]}_B{batch_n}'
                for c_i in range(sampler.get_samples(True)[k_i].shape[0]): #chain, N_steps, 1, N_sys_per_batch
                    key_label = k_i.split('_B')[0]
                    for z_i in range(sampler.get_samples(True)[k_i].shape[-1]):
                        if (key_label!='P_tau') and z_i>=100: continue
                        if (key_label=='P_tau') and z_i>=2000: continue
                        n_z_i = int(z_i + np.sum(N_sys_per_batch[:batch_n])) #System number, starting at 0 for first batch, and for later batches starting at the number of previous systems in earlier batches. Batch number starts at 0.
                        sample_dict[f'{key_label}_{n_z_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0,z_i]
        else: 
            print(k_i,sampler.get_samples(True)[k_i].shape)
            if k_i not in ['Ok','zL','zS']: 
                if k_i=='Ok': assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==3
                if k_i in ['zL','zS']: assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==4
            for c_i in range(sampler.get_samples(True)[k_i].shape[0]):
                try:
                    if k_i=='alpha_weights':
                        for component_i in range(sampler.get_samples(True)[k_i].shape[2]):
                            sample_dict[f'{k_i}_{component_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,component_i] #[(N_chains, N_steps, N_GMM_components)]
                        # print('ALPHA WEIGHTS:',)
                    elif k_i not in ['zL','zS','P_tau','r_theory_2']:
                        sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0]
                        print('Saved shape 1:',k_i,c_i,sampler.get_samples(True)[k_i][c_i,:,0].shape)
                    #May require this if using photometric redshifts
                    else:
                        #print('Shape 2.0:',sampler.get_samples(True)[k_i].shape) #(2, 100, 1, 2048) #chain, N_steps, 1, N_sys
                        for z_i in range(sampler.get_samples(True)[k_i].shape[-1]):
                            if z_i>=100 and k_i!='P_tau': 
                                print(f'Skipping {k_i}_{z_i}_{c_i}')
                                continue #Only saving the first 100 redshifts
                            if z_i>2000 and k_i =='P_tau':
                                print(f'Skipping {k_i}_{z_i}_{c_i}')
                                continue
                            sample_dict[f'{k_i}_{z_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0,z_i]
                            print('Saved shape 2:',f'{k_i}_{z_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:,0,z_i].shape)
                except Exception as ex_1:
                    #May require this exception if using FlatwCDM or FlatLambdaCDM
                    print('Exception here',ex_1,k_i,cosmo_type)
                    try:
                        assert (k_i in ['w_zL','w_zS']) or \
                            (k_i=='Ode') or \
                            (k_i=='Ok' and cosmo_type in ['FlatLambdaCDM','FlatwCDM']) or \
                            (k_i in ['w','wa'] and cosmo_type in ['LambdaCDM','FlatLambdaCDM']) or \
                            (k_i in ['wa'] and wa_const) or (k_i in ['w'] and w0_const) or \
                            (k_i in ['mu_zS_g_L_A','sigma_zS_g_L_A','mu_zS_g_L_B','sigma_zS_g_L_B',
                                        'mu_zL_g_L_A','sigma_zL_g_L_A','mu_zL_g_L_B','sigma_zL_g_L_B'] and fixed_GMM)
                        sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:]
                        print('Saved shape 3:',f'{k_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:].shape)
                    except Exception as ex_0:
                        print(ex_0)
                        print(f'Last-resort saving, assuming non-default initialisation: {k_i},{c_i}')
                        if k_i not in ['zL','zS']: 
                            sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:]
                            print('Saved shape 4:',f'{k_i}_{c_i}',sample_dict[f'{k_i}_{c_i}'].shape)
                        else:
                            for z_i in range(sampler.get_samples(True)[k_i].shape[-1]):
                                if z_i>=100: continue #Only saving the first 100 redshifts
                                sample_dict[f'{k_i}_{z_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,z_i]
                                print('Saved shape 5',f'{k_i}_{z_i}_{c_i}',sample_dict[f'{k_i}_{z_i}_{c_i}'].shape)
    return sample_dict