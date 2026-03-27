# =============================================================================
# SUPPORT PARAMETERIZATION
# =============================================================================
import tensorflow as tf

class SupportParam(tf.keras.layers.Layer):
    """Abstract support parameterization."""
    def initialize(self, batch_size, grid_shape):
        raise NotImplementedError

    def compute_support(self):
        raise NotImplementedError


class HalfSpaceSupport(SupportParam):
    """
    Support defined as intersection of N half-spaces.

    S(x) = ∏_i sigmoid((d_i - n_i · x)/eps)
    """

    def __init__(self, N: int, size_factor: float = 4.0, eps: float = 0.4):
        super().__init__()
        self.N = int(N)
        self.size_factor = float(size_factor)
        self.eps_init = eps

    def initialize(self, batch_size, grid_shape):
        D, H, W = grid_shape

        z = tf.linspace(-(D-1)/2, (D-1)/2, D)
        y = tf.linspace(-(H-1)/2, (H-1)/2, H)
        x = tf.linspace(-(W-1)/2, (W-1)/2, W)

        zz, yy, xx = tf.meshgrid(z, y, x, indexing='ij')
        coords = tf.stack([xx, yy, zz], axis=-1)
        self.coords = coords[None, ...]

        R = float(min(grid_shape)) / self.size_factor

        n0 = tf.math.l2_normalize(
            tf.random.normal([batch_size, self.N, 3]), axis=-1
        )
        d0 = tf.ones([batch_size, self.N]) * R

        self.eps = self.add_weight(
            shape=(), initializer=tf.constant_initializer(self.eps_init),
            trainable=False
        )

        self.n = self.add_weight(
            shape=n0.shape,
            initializer=tf.constant_initializer(n0),
            trainable=True,
            name="normals"
        )

        self.d = self.add_weight(
            shape=d0.shape,
            initializer=tf.constant_initializer(d0),
            trainable=True,
            name="offsets"
        )

    def compute_support(self):
        coords = self.coords[..., None, :]
        normals = self.n[:, None, None, None, :]

        dot = tf.reduce_sum(coords * normals, axis=-1)
        d = self.d[:, None, None, None, :]

        masks = tf.sigmoid((d - dot) / self.eps)

        logS = tf.reduce_sum(
            tf.math.log(tf.clip_by_value(masks, 1e-6, 1.0)), axis=-1
        )
        return tf.exp(logS)

    def project_tangent(self, g):
        inner = tf.reduce_sum(self.n * g, axis=-1, keepdims=True)
        return g - inner * self.n

    def retract(self):
        self.n.assign(tf.math.l2_normalize(self.n, axis=-1))