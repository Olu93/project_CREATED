import pathlib
from pydoc import classname
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils
import tensorflow as tf
from thesis_commons.functions import sample
from thesis_commons import metric

from typing import Generic, Type, TypeVar, NewType
import thesis_generators.models.model_commons as commons

DEBUG_LOSS = False
DEBUG_SHOW_ALL_METRICS = True


# https://keras.io/examples/generative/conditional_gan/
# TODO: Implement an LSTM version of this
class MultiTrainer(models.Model):

    def __init__(self, GeneratorModel: Type[commons.TensorflowModelMixin], *args, **kwargs):
        super(MultiTrainer, self).__init__(name="_".join([cl.__name__ for cl in [type(self), GeneratorModel]]))
        # Seperately trained
        self.max_len = kwargs.get('max_len')
        self.feature_len = kwargs.get('feature_len')
        self.embed_dim = kwargs.get('embed_dim')
        self.vocab_len = kwargs.get('vocab_len')
        self.in_events = layers.Input(shape=(self.max_len, ))
        self.in_features = layers.Input(shape=(self.max_len, self.feature_len))
        self.sampler = commons.Sampler()
        print("Instantiate generator...")
        self.generator = GeneratorModel(*args, **kwargs)


    def compile(self, g_optimizer=None, g_loss=None, g_metrics=None, g_loss_weights=None, g_weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        self.generator.compile(optimizer=g_optimizer or self.generator.optimizer or tf.keras.optimizers.Adam(),
                               loss=g_loss,
                               metrics=g_metrics,
                               loss_weights=g_loss_weights,
                               weighted_metrics=g_weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
        # default_metrics = [metric.MSpCatAcc(name="cat_acc"), metric.MEditSimilarity(name="ed_sim")]

        return super().compile(optimizer=tf.keras.optimizers.Adam(),  run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, **kwargs)



    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = self.generator.call([events, features])
        return rec_ev, rec_ft
    
    # def save(self, *args, **kwargs):
    #     curr_file_path = pathlib.Path(args[0]) 
    #     args = (str(curr_file_path / "trainer"),) + args[1:]
    #     args_generator = (str(curr_file_path / "generator"),) + args[1:]
        
    #     self.generator.save(*args_generator, **kwargs)
    #     return self.save(*args, **kwargs)
    
    # def save(self, *args, **kwargs):
    #     curr_file_path = pathlib.Path(args[0]) 
    #     args = (str(curr_file_path / "trainer"),) + args[1:]
    #     args_generator = (str(curr_file_path / "generator"),) + args[1:]
        
    #     self.generator.save(*args_generator, **kwargs)
    #     return self.save(*args, **kwargs)
    
    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas
    


    def get_generator(self) -> models.Model:
        return self.generator
    



