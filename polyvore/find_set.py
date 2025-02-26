import os
import json

import tensorflow as tf
import pickle as pkl
import numpy as np
import configuration
import polyvore_model_bi as polyvore_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("json_file", "data/label/musinsa.json",
                       "Json file containing the inference data.")
tf.flags.DEFINE_string("image_dir", "data/img",
                       "Directory containing images.")
# tf.flags.DEFINE_string("feature_file", "data/features/test_features_musinsa.pkl",
#                        "Directory to save the features")
tf.flags.DEFINE_string("rnn_type", "", "Type of RNN.")


def main(_):
#   if os.path.isfile(FLAGS.feature_file):
#     print("Feature file already exist.")
#     return
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model_config.rnn_type = FLAGS.rnn_type
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
  g.finalize()
  sess = tf.Session(graph=g)
  saver.restore(sess, FLAGS.checkpoint_path)
  test_json = json.load(open(FLAGS.json_file))
  k = 0

  # Save image ids and features in a dictionary.
  test_features = dict()

  for image_set in test_json:
    set_id = image_set["set_id"]
    image_feat = []
    image_rnn_feat = []
    ids = []
    k = k + 1
    print(str(k) + " : " + set_id)
    for image in image_set["items"]:
      filename = os.path.join(FLAGS.image_dir, 
                              str(image["index"]) + ".jpg")
      with tf.gfile.GFile(filename, "r") as f:
        image_feed = f.read()

      [feat, rnn_feat] = sess.run([model.image_embeddings,
                                   model.rnn_image_embeddings],
                                  feed_dict={"image_feed:0": image_feed})
      
      image_name = set_id + "_" + str(image["index"])
      test_features[image_name] = dict()
      test_features[image_name]["image_feat"] = np.squeeze(feat)
      test_features[image_name]["image_rnn_feat"] = np.squeeze(rnn_feat)
  
#   with open(FLAGS.feature_file, "wb") as f:
#     pkl.dump(test_features, f)

  # Calculate compatibility score.
  compatibility_score = 0.0
  for image_set in test_json:
    set_id = image_set["set_id"]
    for image in image_set["items"]:
      image_name = set_id + "_" + str(image["index"])
      compatibility_score += np.dot(test_features[image_name]["image_feat"],
                                     test_features[image_name]["image_rnn_feat"])
  compatibility_score /= len(test_json)

  print("Compatibility score:", compatibility_score)

if __name__ == "__main__":
  tf.app.run()  