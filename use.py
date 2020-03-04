import tensorflow as tf
import tensorflow_hub as hub

import os
import numpy as np


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        
        self.build_graph(module_url)
        self._warmup()
        

    def build_graph(self, module_url):
        #return
        with self.graph.as_default():
            
            self.embed = hub.Module(module_url)
            self.sts_input1 = tf.placeholder(tf.string, shape=(None))
            self.sts_input2 = tf.placeholder(tf.string, shape=(None))

            sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
            sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
            self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
            clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
            self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
            self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            
    def _warmup(self):
        sample_text = "sample_text"
        _ = self.semantic_sim(sample_text, sample_text)

    def semantic_sim(self, sents1, sents2):
        #scores = [np.ones(len(sents1))-0.03]
        #return scores
        
        
        with self.sess.as_default():
            scores = self.sess.run(
                [self.sim_scores],
                feed_dict={
                    self.sts_input1: np.atleast_1d(sents1),
                    self.sts_input2: np.atleast_1d(sents2),
                })
            return scores