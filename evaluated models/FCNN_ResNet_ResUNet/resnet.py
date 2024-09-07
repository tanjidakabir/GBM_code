import tensorflow as tf 
import numpy as np 
import os
import time
import SimpleITK as sitk
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="10"


from fcnn2d import FCNN_2D, Base
from layers import *
from auxiliary import *
from pre_process import remove_background_3d_with_label
from image2array import single2Image, save_pred, five2four


class RESNET(FCNN_2D):

	def build(self, X, y):
		# images = (X_train - tf.reduce_mean(X_train, axis = 0, keep_dims = True)) / tf.reduce_max(X_train, axis = 0, keep_dims = True)

		# with tf.device('gpu:1'):

		with tf.variable_scope('conv_1'):
			conv_1 = conv_layer_res(X, [7, 7, 4, 64], [1, 1, 1, 1], 'conv_1_1')
			bn_1 = tf.nn.relu(bn_layer(conv_1, self.is_training, 'bn_1'))
			max_1 = max_pool_layer(bn_1, [1, 3, 3, 1], [1, 2, 2, 1], name = 'max_1')

		with tf.variable_scope('bottleneck_1'):
			bottleneck_1_1 = bottle_layer(max_1, 64, 64, 128, self.is_training, 'bottle_1')
			bottleneck_1_2 = bottle_layer(bottleneck_1_1, 128, 64, 128, self.is_training, 'bottle_2')
			bottleneck_1_3 = bottle_layer(bottleneck_1_2, 128, 64, 128, self.is_training, 'bottle_3')

		with tf.variable_scope('bottleneck_2'):
			max_2 = max_pool_layer(bottleneck_1_3, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_2')
			bottleneck_2_1 = bottle_layer(max_2, 128, 128, 256, self.is_training, 'bottle_1')
			bottleneck_2_2 = bottle_layer(bottleneck_2_1, 256, 128, 256, self.is_training, 'bottle_2')
			bottleneck_2_3 = bottle_layer(bottleneck_2_2, 256, 128, 256, self.is_training, 'bottle_3')
			bottleneck_2_4 = bottle_layer(bottleneck_2_3, 256, 128, 256, self.is_training, 'bottle_4')

		# with tf.device('gpu:2'):

		with tf.variable_scope('bottleneck_3'):
			max_3 = max_pool_layer(bottleneck_2_4, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_3')
			bottleneck_3_1 = bottle_layer(max_3, 256, 256, 1024, self.is_training, 'bottle_1')
			bottleneck_3_2 = bottle_layer(bottleneck_3_1, 1024, 256, 1024, self.is_training, 'bottle_2')
			bottleneck_3_3 = bottle_layer(bottleneck_3_2, 1024, 256, 1024, self.is_training, 'bottle_3')
			bottleneck_3_4 = bottle_layer(bottleneck_3_3, 1024, 256, 1024, self.is_training, 'bottle_4')
			bottleneck_3_5 = bottle_layer(bottleneck_3_4, 1024, 256, 1024, self.is_training, 'bottle_5')
			bottleneck_3_6 = bottle_layer(bottleneck_3_5, 1024, 256, 1024, self.is_training, 'bottle_6')

		with tf.variable_scope('bottleneck_4'):
			max_4 = max_pool_layer(bottleneck_3_6, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_4')
			bottleneck_4_1 = bottle_layer(max_4, 1024, 512, 2048, self.is_training, 'bottle_1')
			bottleneck_4_2 = bottle_layer(bottleneck_4_1, 2048, 512, 2048, self.is_training, 'bottle_2')
			bottleneck_4_3 = bottle_layer(bottleneck_4_2, 2048, 512, 2048, self.is_training, 'bottle_3')

		max_5 = max_pool_layer(bottleneck_4_3, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_5')

		fc_1 = tf.nn.dropout(conv_layer_res(max_5, [1, 1, 2048, 2048], [1, 1, 1, 1], 'fc_1'), self.dropout)
		fc_2 = conv_layer_res(fc_1, [1, 1, 2048, self.num_classes], [1, 1, 1, 1], 'fc_2')

		# Now we start upsampling and skip layer connections.
		img_shape = tf.shape(X)
		dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])
		upsample_1 = upsample_layer(fc_2, dconv3_shape, self.num_classes, 'upsample_1', 32)

		skip_1 = skip_layer_connection(max_4, 'skip_1', 1024, stddev=0.00001)
		upsample_2 = upsample_layer(skip_1, dconv3_shape, self.num_classes, 'upsample_2', 16)

		skip_2 = skip_layer_connection(max_3, 'skip_2', 256, stddev=0.0001)
		upsample_3 = up_layer(skip_2, dconv3_shape, 5, 5, 8, 'upsample_3')


		logits = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
		self.result = tf.argmax(logits, axis = 3)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix



if __name__ == '__main__':
	print ('loading from HGG_train.npz...')
	#f = np.load(Base + '/HGG_train2.npz')
	f = np.load('HGG_train_only_45_patients.npz')
	print('DEBUG 5',f.files)
	X = f['arr_0']#f['X']
	y = f['arr_1']#f['y']

	print ("DEBUG 6",X.shape, y.shape)	
	

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	#net = VGG_DICE(input_shape = (240, 240, 4), num_classes = 5)
	net = RESNET(input_shape = (240, 240, 4), num_classes = 5)
	print('DEBUG 7', "Model constructor build")

	

	model_name = 'model_resnet_1_99'
	pred = net.predict(model_name, X).astype('uint8')
	#pred = five2four(pred)
	print("DEBUG 10", pred.shape)

	temp_image = pred[0,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '35256564'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[1,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '47303976'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[2,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '40552846'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[3,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34114852'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[4,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '61051121'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[5,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '85322558'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[6,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '45865500'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[7,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39621144'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[8,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46674661'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[9,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34116070'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[10,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '38469985'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[11,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39991662'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)



	temp_image = pred[12,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46140963'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[13,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '61043683'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[14,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34290186'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[15,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39823718'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[16,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '85405196'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[17,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34804884'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[18,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '33929672'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[19,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39618302'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[20,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39925316'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[21,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46741565'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[22,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39981533'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[23,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '36477034'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[24,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '39597056'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[25,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '35520522'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[26,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '47315835'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[27,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '36104836'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[28,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '40481407'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[29,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34012715'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[30,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '45947450'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[31,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '38216722'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[32,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '40653045'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[33,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '38845165'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[34,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '38401171'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[35,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '85145554'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	
	temp_image = pred[36,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '35525486'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[37,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '45432352'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[38,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '45415776'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[39,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46446682'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	

	temp_image = pred[40,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46292941'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[41,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34259182'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[42,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '34169940'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[43,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '45533978'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


	temp_image = pred[44,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)
	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '46481193'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)


















	'''



	temp_image = pred[1,:,:,:]
	temp_image = np.where(temp_image==4,0, temp_image)
	#temp_image=np.where(temp_image>.5, 1,0)


	out_nii = os.path.join('./prediction/'+model_name+'/HGG_train', '38216722'+'.nii.gz')
	#sitk_img = sitk.GetImageFromArray(pred[0,:,:,:])
	sitk_img = sitk.GetImageFromArray(temp_image)
	sitk.WriteImage(sitk_img , out_nii)

	'''

	#save_pred(pred, './prediction/'+model_name+'/HGG_train', 'HGG_train.json')

	

	#net.train(X, y, model_name = 'model_resnet_1', train_mode = 0, batch_size = 4, learning_rate = 5e-5, epoch = 1, restore = False, N_worst = 1e10, thre = 1.0)






	#net.multi_gpu_train(X, y, model_name = 'model_resnet_1', train_mode = 1, num_gpu = 1, batch_size = 4, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e10, thre = 1.0)
 
	# else:
	# 	exit(0)
