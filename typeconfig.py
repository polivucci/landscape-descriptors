import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
default_dtype = jnp.float64
# sanity check
assert jnp.array(1.0, dtype=jnp.float64).dtype == jnp.float64, \
    "jax_enable_x64 was not set in time"

# default_dtype = jnp.float32