import jax.numpy as jnp
import jax


def make_choice(choices):
    t = jnp.array([30])  # example input
    conditions = [
        (t > 0) & (t < 60),
        (t > 60) & (t < 120),
        (t > 120)
    ]
    result = jnp.select(conditions, choices, default=0)
    return result
choices = [1, 2, 3]
jit_make_choice = jax.jit(make_choice)
print(jit_make_choice(choices))  # [1 2 3 0]