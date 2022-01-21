import os
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
import optax
import jmp
import time

from funlib.learn.jax.models import UNet, ConvPass

from typing import Tuple, Any, NamedTuple, Dict

# some JAX installations require this to run properly
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'


'''To test model with some dummy input and output, run with command

    `CUDA_VISIBLE_DEVICES=0 python unet_example.py`

for single device training, or

    `CUDA_VISIBLE_DEVICES=0,1,2,3 python unet_example.py`

for multi-device training
'''

# PARAMETERS
mp_training = True  # mixed-precision training using `jmp`
learning_rate = 0.5e-4
pmap_sum_grads = False  # training seems to be faster with grads summing


class Params(NamedTuple):
    weight: jnp.ndarray
    opt_state: jnp.ndarray
    loss_scale: jmp.LossScale


# should be the same as gunpowder.jax.GenericJaxModel
# replicated here to reduce dependency
class GenericJaxModel():

    def __init__(self):
        pass

    def initialize(self, rng_key, inputs, is_training):
        raise RuntimeError("Unimplemented")

    def forward(self, inputs):
        raise RuntimeError("Unimplemented")

    def train_step(self, inputs, pmapped):
        raise RuntimeError("Unimplemented")


class Model(GenericJaxModel):

    def __init__(self):
        super().__init__()

        # we encapsulate the UNet and the ConvPass in one hk.Module
        # to make assigning precision policy easier
        class MyModel(hk.Module):

            def __init__(self, name=None):
                super().__init__(name=name)
                self.unet = UNet(
                    num_fmaps=24,
                    fmap_inc_factor=3,
                    downsample_factors=[[2,2,2],[2,2,2],[2,2,2]],
                    )
                self.conv = ConvPass(
                    kernel_sizes=[[1,1,1]],
                    out_channels=3,
                    activation='sigmoid',
                    )

            def __call__(self, x):
                return self.conv(self.unet(x))

        def _forward(x):
            net = MyModel()
            return net(x)

        if mp_training:
            policy = jmp.get_policy('p=f32,c=f16,o=f32')
        else:
            policy = jmp.get_policy('p=f32,c=f32,o=f32')
        hk.mixed_precision.set_policy(MyModel, policy)

        self.model = hk.without_apply_rng(hk.transform(_forward))
        self.opt = optax.adam(learning_rate)

        @jit
        def _forward(params, inputs):
            return {'affs': self.model.apply(params.weight, inputs['raw'])}

        self.forward = _forward

        @jit
        def _loss_fn(weight, raw, gt, mask, loss_scale):
            pred_affs = self.model.apply(weight, x=raw)
            loss = optax.l2_loss(predictions=pred_affs, targets=gt)
            loss = loss*2*mask  # optax divides loss by 2 so we mult it back
            loss_mean = loss.mean(where=mask)
            return loss_scale.scale(loss_mean), (pred_affs, loss, loss_mean)

        @jit
        def _apply_optimizer(params, grads):
            updates, new_opt_state = self.opt.update(grads, params.opt_state)
            new_weight = optax.apply_updates(params.weight, updates)
            return new_weight, new_opt_state

        def _train_step(params, inputs, pmapped=False) -> Tuple[Params, Dict[str, jnp.ndarray], Any]:

            raw, gt, mask = inputs['raw'], inputs['gt'], inputs['mask']

            grads, (pred_affs, loss, loss_mean) = jax.grad(_loss_fn, has_aux=True)(
                params.weight, raw, gt, mask, params.loss_scale)

            # dynamic mixed precision loss scaling
            grads = params.loss_scale.unscale(grads)
            if pmapped:
                # combine grads, but cast to compute precision (f16) first
                grads = policy.cast_to_compute(grads)
                grads = params.loss_scale.unscale(grads)
                if pmap_sum_grads:
                    grads = jax.lax.psum(grads, axis_name='num_devices')
                else:
                    grads = jax.lax.pmean(grads, axis_name='num_devices')
                grads = policy.cast_to_param(grads)
            new_weight, new_opt_state = _apply_optimizer(params, grads)

            # skip nonfinite updates (https://github.com/deepmind/jmp)
            grads_finite = jmp.all_finite(grads)
            new_loss_scale = params.loss_scale.adjust(grads_finite)
            new_weight, new_opt_state = jmp.select_tree(grads_finite,
                                                        (new_weight, new_opt_state),
                                                        (params.weight, params.opt_state))

            new_params = Params(new_weight, new_opt_state, new_loss_scale)
            outputs = {'affs': pred_affs, 'grad': loss}
            return new_params, outputs, loss_mean

        self.train_step = _train_step

    def initialize(self, rng_key, inputs, is_training=True):
        weight = self.model.init(rng_key, inputs['raw'])
        opt_state = self.opt.init(weight)
        if mp_training:
            loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15))
        else:
            loss_scale = jmp.NoOpLossScale()
        return Params(weight, opt_state, loss_scale)


def split(arr, n_devices):
    """Splits the first axis of `arr` evenly across the number of devices."""
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def create_network():
    # returns a model that Gunpowder `Predict` and `Train` node can use
    return Model()


if __name__ == "__main__":

    my_model = Model()

    n_devices = jax.local_device_count()
    batch_size = 4*n_devices

    raw = jnp.ones([batch_size, 1, 132, 132, 132])
    gt = jnp.zeros([batch_size, 3, 40, 40, 40])
    mask = jnp.ones([batch_size, 3, 40, 40, 40])
    inputs = {
        'raw': raw,
        'gt': gt,
        'mask': mask,
    }
    rng = jax.random.PRNGKey(42)

    # init model
    if n_devices > 1:
        # split input for pmap
        raw = split(raw, n_devices)
        gt = split(gt, n_devices)
        mask = split(mask, n_devices)
        single_device_inputs = {
            'raw': raw,
            'gt': gt,
            'mask': mask,
        }
        rng = jnp.broadcast_to(rng, (n_devices,) + rng.shape)
        model_params = jax.pmap(my_model.initialize)(rng, single_device_inputs)

    else:
        model_params = my_model.initialize(rng, inputs, is_training=True)

    # test forward
    y = jit(my_model.forward)(model_params, {'raw': raw})
    assert y['affs'].shape == (batch_size, 3, 40, 40, 40)

    # test train loop
    for _ in range(10):
        t0 = time.time()

        if n_devices > 1:
            model_params, outputs, loss = jax.pmap(
                                my_model.train_step,
                                axis_name='num_devices',
                                donate_argnums=(0,),
                                static_broadcasted_argnums=(2,))(
                                model_params, inputs, True)
        else:
            model_params, outputs, loss = jax.jit(
                                my_model.train_step,
                                donate_argnums=(0,),
                                static_argnums=(2,))(
                                model_params, inputs, False)

        print(f'Loss: {loss}, took {time.time()-t0}s')
