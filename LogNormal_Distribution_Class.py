import jax.numpy as jnp

class jax_lognormal:
    def __init__(self,x,loc,scale,s):
        # assert x>loc
        self.y = (x-loc)/scale
        self.scale = scale
        self.s = s
    def pdf(self):
        return (1/((self.s*self.y*jnp.sqrt(2*jnp.pi))))*jnp.exp(-0.5*(jnp.log(self.y)/self.s)**2)/self.scale
    def log_prob(self):
        return jnp.log((1/((self.s*self.y*jnp.sqrt(2*jnp.pi)))))-0.5*(jnp.log(self.y)/self.s)**2 - jnp.log(self.scale)


# LN_dict = {'x': 6.21634899371342, 'loc': -0.0007176, 'scale': 0.10081474, 's': 0.10007405}
# TN_dict = {'a': -5, 'b': np.inf, 'x': -6, 'loc': 5.5, 'scale': 1}

# LN_LP = jax_lognormal(**LN_dict).log_prob()
# LN_P = jax_lognormal(**LN_dict).pdf()

# TN_LP = jax_truncnorm.logpdf(**TN_dict)
# TN_P = jax_truncnorm.pdf(**TN_dict)

# print(LN_LP,TN_LP)
# print(LN_P,TN_P)
# print(np.exp(LN_LP+TN_LP),LN_P*TN_P)