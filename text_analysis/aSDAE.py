'''
created on July 15, 2017

@author: Beili
'''

import numpy as np

from keras.layers import Input,Dense,concatenate
from keras.engine import Model
from keras.preprocessing import sequence
from keras import backend as K

class aSDAE_module():
  
    batch_size=128
    epochs=8
    
    def feature_loss(self,y_true,y_pred):
        return K.mean(K.square(y_true-y_pred),axis=-1)

    def __init__(self,first_dimension,output_dimension,item_num,user_feature):
    
        self.maxlen=item_num
        self.maxfea=user_feature
        
        model_input_user_rating=Input(shape=[item_num],name='user_rating')
        model_input_user_sideinformation=Input(shape=(user_feature,),name='user_sideinformation')
        
        #model_input_user_rating=model_input_user_rating+0.5*np.random.normal(loc=0,scale=100,size=item_num)
        #model_input_user_sideinformation=model_input_user_sideinformation+0.5*np.random.normal(loc=0,scale=100,size=user_feature)
     
        
        model_input=concatenate([model_input_user_rating,model_input_user_sideinformation])
        
        encoder_1=Dense(first_dimension,activation='relu',name='encoder_1')(model_input)
        #encoder_conc=concatenate([encoder_1,model_input_user_sideinformation])
        
        encoder_2=Dense(output_dimension,activation='relu',name='user_matrix')(encoder_1)
        #decoder_conc=concatenate([encoder_2,model_input_user_sideinformation])
        
        decoder_3=Dense(first_dimension,activation='relu',name='decoder_1')(encoder_2)
        #decoder_conc=concatenate([decoder_3,model_input_user_sideinformation])
        
        model_output_user_rating=Dense(item_num,activation='sigmoid',name='output_model_rating')(decoder_3)
        model_output_user_sideinformation=Dense(user_feature,activation='sigmoid',name='output_model_side')(decoder_3)
        
        output_model=Model(inputs=[model_input_user_rating,model_input_user_sideinformation],outputs=[model_output_user_rating,model_output_user_sideinformation,encoder_2])
        output_model.compile(optimizer='rmsprop',loss={'output_model_rating':'mse','output_model_side':'mse','user_matrix':'mse'},loss_weights=[1,1,0])
        self.model=output_model
            
     
    def train(self,aSDAE,train_user,U,seed):
    
        #aSDAE=sequence.pad_sequences(aSDAE,maxlen=self.maxlen)
        np.random.seed(seed)
        aSDAE=np.random.permutation(aSDAE)
        
        #train_user=sequence.pad_sequences(train_user,maxlen=self.maxfea)
        np.random.seed(seed)
        train_user=np.random.permutation(train_user)
        
        np.random.seed(seed)
        U=np.random.permutation(U)
        
        print("Train... aSDAE module")
        self.model.compile(optimizer='rmsprop',loss={'output_model_rating':'mse','output_model_side':'mse','user_matrix':'mse'},loss_weights=[1,1,1])
        history=self.model.fit({'user_rating':aSDAE,'user_sideinformation':train_user},{'output_model_rating':aSDAE,'output_model_side':train_user,'user_matrix':U},verbose=0,batch_size=self.batch_size,epochs=self.epochs)
    
        return history

    def get_middle_layer(self,aSDAE,train_user):

        #aSDAE=sequence.pad_sequences(aSDAE,maxlen=self.maxlen)
        #train_user=sequence.pad_sequences(train_user,maxlen=self.maxfea)
        middle=self.model.predict({'user_rating':aSDAE,'user_sideinformation':train_user},batch_size=self.batch_size)[2]
        
        return middle
    
    def load_model(self, model_path):
        self.model.load_weights(model_path)
    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)   
