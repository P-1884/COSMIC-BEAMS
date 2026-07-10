import jax.numpy as jnp

class jax_lognormal:
    def __init__(self,x,loc,scale,s):
        self.y = (x-loc)/scale
        self.scale = scale
        self.s = s
    def pdf(self):
        return (1/((self.s*self.y*jnp.sqrt(2*jnp.pi))))*jnp.exp(-0.5*(jnp.log(self.y)/self.s)**2)/self.scale
    def log_prob(self):
        return jnp.log((1/((self.s*self.y*jnp.sqrt(2*jnp.pi)))))-0.5*(jnp.log(self.y)/self.s)**2 - jnp.log(self.scale)
