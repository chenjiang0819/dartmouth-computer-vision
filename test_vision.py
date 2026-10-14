import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from vision import load_images,fit_svm,preprocess_rgb,generators,build_model
class Checks(unittest.TestCase):
 def test_mapping_types_and_svm(self):
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
   train=Path(d)/'train';test=Path(d)/'test'
   for root,names in [(train,['z','a']),(test,['z'])]:
    for name in names:
     (root/name).mkdir(parents=True);Image.new('RGBA',(3,4),(255 if name=='z' else 0,0,0,255)).save(root/name/'a.png')
   X,y,classes=load_images(train);self.assertEqual(classes,['a','z']);self.assertEqual(y.tolist(),[0,1])
   Xt,yt,_=load_images(test,classes);self.assertEqual(yt.tolist(),[1]);self.assertEqual(fit_svm(X,y).predict(Xt).tolist(),[1])
   self.assertEqual(fit_svm(X,y,kernel='linear').predict(Xt).tolist(),[1])
   with self.assertRaises(ValueError):load_images(test,['../z'])
   (test/'unknown').mkdir()
   with self.assertRaises(ValueError):load_images(test,classes)
 def test_preprocessing_oracle(self):
  x=np.array([0,127.5,255],dtype=np.float32).reshape(1,1,1,3)
  np.testing.assert_allclose(preprocess_rgb(x,True),[[[[-1,0,1]]]])
  np.testing.assert_allclose(preprocess_rgb(x),[[[[0,.5,1]]]])
  with self.assertRaises(ValueError):preprocess_rgb(np.full((1,1,1,3),np.nan))
 def test_generators_split_alignment(self):
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
   train=Path(d)/'train';test=Path(d)/'test'
   for root,names,count in [(train,['z','a','b'],5),(test,['z','b'],2)]:
    for name in names:
     (root/name).mkdir(parents=True)
     for i in range(count):Image.new('RGB',(71,71),(i*30,0,0)).save(root/name/f'{i}.png')
   tr,va,te=generators(train,test)
   self.assertEqual(tr.class_indices,te.class_indices);self.assertEqual(te.classes.tolist(),[1,1,2,2]);self.assertFalse(te.shuffle)
   self.assertFalse(set(tr.filenames)&set(va.filenames));self.assertEqual(tr.samples+va.samples,15)
 def test_all_architectures_loss(self):
  import tensorflow as tf
  tf.keras.utils.set_random_seed(0)
  x=np.zeros((2,71,71,3),np.float32);y=np.array([0,1])
  for kind in ['mlp','cnn','custom','xception']:
   model=build_model(kind,3,weights=None)
   probs=model(x,training=False).numpy();self.assertEqual(probs.shape,(2,3));np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
   loss=model.train_on_batch(x,y);self.assertTrue(np.isfinite(loss).all())
   if kind=='xception':self.assertFalse(model.layers[0].trainable)
   tf.keras.backend.clear_session()
if __name__=='__main__':unittest.main()
