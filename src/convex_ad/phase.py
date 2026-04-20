# =============================================================================
# PHASE PARAMETERIZATION
# =============================================================================
import tensorflow as tf

class PhaseParam(tf.keras.layers.Layer):
    def initialize(self, batch_size, grid_shape, initial_guess=None):
        raise NotImplementedError


class GridPhase(PhaseParam):
    """Direct phase parameterization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = None
        self._grid_shape = None
        self._initial_guess = None

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        self._batch_size = batch_size
        self._grid_shape = grid_shape
        self._initial_guess = initial_guess
        self.build()

    def build(self, input_shape=None):
        if self._grid_shape is None:
            return
        if self._initial_guess is not None:
            phase = tf.math.angle(self._initial_guess)
            if len(phase.shape) == 3:
                phase = phase[None, ...]
            initializer = tf.keras.initializers.Constant(phase.numpy())
            shape = phase.shape
        else:
            initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            shape = [self._batch_size, *self._grid_shape]
        self.phase = self.add_weight(
            name="phase", shape=shape, dtype=tf.float32,
            trainable=True, initializer=initializer,
        )
        super().build(input_shape)

    def compute_phase(self):
        return self.phase


class GridPhasor(PhaseParam):
    """Unit-circle phasor parameterization."""

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self._batch_size = None
        self._grid_shape = None

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        self._batch_size = batch_size
        self._grid_shape = grid_shape
        self.build()

    def build(self, input_shape=None):
        if self._grid_shape is None:
            return
        shape = [self._batch_size, *self._grid_shape]
        self.c_raw = self.add_weight(
            name="c_raw", shape=shape, dtype=tf.float32,
            trainable=True, initializer=tf.keras.initializers.Ones(),
        )
        self.s_raw = self.add_weight(
            name="s_raw", shape=shape, dtype=tf.float32,
            trainable=True, initializer=tf.keras.initializers.Zeros(),
        )
        super().build(input_shape)

    def compute_phasor(self):
        denom = tf.sqrt(self.c_raw**2 + self.s_raw**2 + self.eps)
        return self.c_raw / denom, self.s_raw / denom
