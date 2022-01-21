import pytest
import unittest
import warnings
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
from funlib.learn.jax.models import UNet
warnings.filterwarnings("error")


class TestUNet(unittest.TestCase):

    def test_creation(self):

        x = jnp.zeros((1, 1, 100, 80, 48))

        def _forward(x):
            unet = UNet(
                num_fmaps=3,
                fmap_inc_factor=2,
                downsample_factors=[[2, 2, 2], [2, 2, 2]])
            return unet(x)
        model = hk.without_apply_rng(hk.transform(_forward))
        rng_key = jax.random.PRNGKey(42)
        weight = model.init(rng_key, x)

        y = jit(model.apply)(weight, x)

        assert y.shape == (1, 3, 60, 40, 8)

        def _forward(x):
            unet = UNet(
                num_fmaps=3,
                fmap_inc_factor=2,
                downsample_factors=[[2, 2, 2], [2, 2, 2]],
                num_fmaps_out=5)
            return unet(x)
        model = hk.without_apply_rng(hk.transform(_forward))
        rng_key = jax.random.PRNGKey(42)
        weight = model.init(rng_key, x)

        y = jit(model.apply)(weight, x)

        assert y.shape == (1, 5, 60, 40, 8)

    def test_shape_warning(self):

        x = jnp.zeros((1, 1, 100, 80, 48))

        # Should raise warning
        with pytest.raises(Exception):

            def _forward(x):
                unet = UNet(
                    num_fmaps=3,
                    fmap_inc_factor=2,
                    downsample_factors=[[2, 3, 2], [2, 2, 2]],
                    num_fmaps_out=5)
                return unet(x)
            model = hk.without_apply_rng(hk.transform(_forward))
            rng_key = jax.random.PRNGKey(42)
            weight = model.init(rng_key, x)
            jit(model.apply)(weight, x)

    # # def test_4d(self):
    #     # TODO

    def test_multi_head(self):

        x = jnp.zeros((1, 1, 100, 80, 48))

        def _forward(x):
            unet = UNet(
                num_fmaps=3,
                fmap_inc_factor=2,
                downsample_factors=[[2, 2, 2], [2, 2, 2]],
                num_heads=3)
            return unet(x)
        model = hk.without_apply_rng(hk.transform(_forward))
        rng_key = jax.random.PRNGKey(42)
        weight = model.init(rng_key, x)

        y = jit(model.apply)(weight, x)

        assert len(y) == 3
        assert y[0].shape == (1, 3, 60, 40, 8)
        assert y[1].shape == (1, 3, 60, 40, 8)
        assert y[2].shape == (1, 3, 60, 40, 8)
