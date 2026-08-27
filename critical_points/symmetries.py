import jax.numpy as jnp
from jax.debug import print as jaxprint


def periodic_sym(x, low, up):
    width = up - low
    return low + jnp.mod(x - low, width)


def shift_true_forward(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Given a flat JAX bool array, shifts all True values forward by one position.
    The value at the previous position becomes False.
    Asserts that the last element must never be True.

    Example: [F, T, F, T, F] -> [F, F, T, F, T]

    Args:
        arr: A 1D JAX boolean array.

    Returns:
        A new boolean array of the same shape with True values shifted forward by 1.
    """
    # assert arr.ndim == 1, "Input must be a flat (1D) array"
    # assert not arr[-1], "The last element of the input array must never be True"

    # Shift right: prepend False, drop the last element
    shifted = jnp.concatenate([jnp.array([False]), arr[:-1]])
    return shifted


def make_periodize_fn(periodize, low, up):
    """Creates a function that makes x periodic between the two given bounds conditional on flag."""

    def periodize_fn(x):
        periodic_x = periodic_sym(x, low, up)
        return jnp.where(periodize, periodic_x, x)

    return periodize_fn


def make_map_0pi_fn(map_0pi, map_phis=None):

    def map_0pi_fn(x):
        """Maps negative amplitudes to positive amplitudes
        (neg amplitude is equivalent to pos amplitude with neg phase).

        Flips sign of negative amplitudes and shift phases by 0.5.
        -A*sin(theta)=A*sin(theta+pi)
        NB Works if [0,1] maps to [0,positive_number] and if phases map to .
        """
        amps = jnp.where(map_0pi, x, 0.0)  # single out amplitudes

        neg_amps = amps < 0.0

        # jaxprint('amps0 {w}', w=amps)

        periodized_02 = periodic_sym(amps, 0.0, 2.0)
        periodized_0m2 = periodic_sym(amps, -2.0, 0.0)
        amps = jnp.where(neg_amps, periodized_0m2, periodized_02)

        # jaxprint('amps1 {w}', w=amps)

        gt1_amps = amps > 1.0  # amps>1
        ltm1_amps = amps < -1.0  # amps<-1
        amps = jnp.where(gt1_amps, amps - 2.0, amps)  # turn amps>1 into neg amps -1,0
        amps = jnp.where(
            ltm1_amps, amps + 2.0, amps
        )  # turn amps<-1 into pos amps in 0,1

        # jaxprint('amps2 {w}', w=amps)

        neg_amps = amps < 0.0
        signs_amps = jnp.where(neg_amps, -1.0, 1.0)  # flip signs of neg amplitudes

        neg_amps_phis = shift_true_forward(neg_amps)
        offset = jnp.where(
            neg_amps_phis, 0.5, 0.0
        )  # add offset of 0.5 (=pi) to the phases as -A*sin(theta)=A*sin(theta+pi)
        # jaxprint('offset {w}', w=offset)

        new_x = jnp.where(map_0pi, amps * signs_amps, x) + offset
        # jaxprint('new_x {w}', w=new_x)

        return new_x

    return map_0pi_fn


def make_evenize_head_fn(evenize, phis):

    def reflect_fn(x):
        """Reflects -x[0] (head amplitude) around zero
        and x[1:] (phase constants) around 0.5 (i.e. 0 if range=[-pi,pi]).
        """
        flip = x[0] < 0
        signs = jnp.where(flip, -1.0, 1.0)
        offset = jnp.where(phis, jnp.where(flip, 1.0, 0.0), 0.0)
        return x * signs + offset

    def evenize_head_fn(x):
        apply = evenize[0] == True
        return jnp.where(apply, reflect_fn(x), x)

    return evenize_head_fn


def make_evenize_fn(evenize):

    # def reflect_fn(x):
    #     '''Reflects -x[0] (head amplitude) around zero
    #     and x[1:] (phase constants) around 0.5 (i.e. 0 if range=[-pi,pi]).
    #     '''
    #     flip = x[0] < 0
    #     signs = jnp.where(flip, -1.0, 1.0)
    #     offset = jnp.where(jnp.arange(x.shape[0]) > 0, jnp.where(flip, 1.0, 0.0), 0.0)
    #     return x * signs + offset

    # def evenize_fn(x):
    #     apply = evenize[0]==True
    #     return jnp.where(apply, reflect_fn(x), x)

    def evenize_fn(x):
        x_pos = jnp.abs(x)
        return jnp.where(evenize, x_pos, x)

    # def evenize_fn(x):
    #     '''Maps negative amplitudes to positive amplitudes
    #     (neg amplitude is equivalent to pos amplitude with neg phase).

    #     Flips sign of negative amplitudes and shift phases by 0.5.
    #     -A*sin(theta)=A*sin(theta+pi)
    #     NB Works if [0,1] maps to [0,positive_number].
    #     '''
    #     amps = jnp.where(evenize, x, 0.0)             # single out amplitudes

    #     neg_amps = amps < 0.0
    #     apply = neg_amps

    #     signs_amps = jnp.where(neg_amps, -1.0, 1.0)   # flip signs of neg amplitudes

    #     neg_amps_phis = shift_true_forward(neg_amps)
    #     offset = jnp.where(neg_amps_phis, 0.5, 0.0)    # add offset of 0.5 (=pi) to the phases as -A*sin(theta)=A*sin(theta+pi)

    #     new_x = jnp.where(evenize, amps*signs_amps, x) + offset

    #     flip = jnp.any(apply) # apply condition; true or false
    #     return jnp.where(flip, new_x, x) # apply symmetry if condition is true

    return evenize_fn


def make_symmetrize_fn(symmetries_fns):
    """Creates a function that makes x symmetric."""

    def symmetrize_fn(x):
        """Composes symmetries on x."""
        for sym_fn in symmetries_fns:
            x = sym_fn(x)
        return x

    return symmetrize_fn
