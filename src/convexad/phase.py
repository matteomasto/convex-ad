# =============================================================================
# PHASE PARAMETERIZATION
# =============================================================================
import tensorflow as tf

class PhaseParam(tf.keras.layers.Layer):
    def initialize(self, batch_size, grid_shape, initial_guess=None):
        raise NotImplementedError


class GridPhase(PhaseParam):
    """Direct phase parameterization."""

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        if initial_guess is not None:
            phase = tf.math.angle(initial_guess)
            if phase.ndim == 3:
                phase = phase[None, ...]
        else:
            phase = tf.random.uniform([batch_size, *grid_shape])

        self.phase = tf.Variable(phase, trainable=True)
        return self.phase

    def compute_phase(self):
        return self.phase


class GridPhasor(PhaseParam):
    """Unit-circle phasor parameterization."""

    def initialize(self, batch_size, grid_shape, initial_guess=None, eps=1e-8):
        self.eps = eps

        shape = [batch_size, *grid_shape]

        c0 = tf.ones(shape)
        s0 = tf.zeros(shape)

        self.c_raw = self.add_weight(shape=shape, initializer=tf.constant_initializer(c0))
        self.s_raw = self.add_weight(shape=shape, initializer=tf.constant_initializer(s0))

    def compute_phasor(self):
        denom = tf.sqrt(self.c_raw**2 + self.s_raw**2 + self.eps)
        return self.c_raw / denom, self.s_raw / denom
