from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer, PReLU
from sklearn.metrics import mean_squared_log_error


def scale_and_encode(data, scaler, encoder, cat_names, num_names):
    numeric = scaler.transform(data[num_names])
    category = encoder.transform(data[cat_names]).toarray().astype(int)
    return np.concatenate((numeric, category), axis=1)

def compile_and_fit_regression_model(train_x, train_y, layers, nodes, epochs, verb):
        
        model = Sequential()

        model.add(Dense(nodes, input_dim=train_x.shape[1], activation='relu'))
        model.add(Dropout(0.1))
        
        for i in range(1,layers):
            model.add(Dense(nodes))
            model.add(Dropout(0.1))
        
        model.add(Dense(1))
    
        model.compile(optimizer='nadam', loss='MSLE', metrics=['mae'])
        model.fit(train_x, train_y, epochs=epochs,verbose=verb)
        return model

def compile_and_fit_classification_model(train_x, train_y, layers, nodes, epochs, verb):
        
        model = Sequential()

        model.add(Dense(nodes, input_dim=train_x.shape[1], activation='relu'))
        model.add(Dropout(0.1))
        
        for i in range(1,layers):
            model.add(Dense(nodes))
            model.add(Dropout(0.1))
        
        model.add(Dense(train_y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.fit(train_x, train_y, epochs=epochs,verbose=verb)
        return model

class MainTensor():
    
    def __init__(self, dataframe, target):
        
        self.dataframe = dataframe
        self.target = target
        self.target_is_category = True if self.dataframe[self.target].dtype.kind == 'O' else False

        y = self.dataframe[[self.target]] 
        x = (self.dataframe).drop(self.target, axis=1)

        # Create list of column names
        self.category_column_names = list(x.select_dtypes(include='object').columns)
        self.numeric_column_names = list(x.select_dtypes(include=['float','int']).columns)

        #Fit one hot encoder to all rows of X
        encoder_x = OneHotEncoder()
        encoder_x.fit(x[self.category_column_names])
        self.encoder_x = encoder_x

        #Encode categorical Y_data
        if self.target_is_category:
            encoder_y = OneHotEncoder()
            y = encoder_y.fit_transform(y).toarray().astype(int)
            self.encoder_y = encoder_y
        else:
            pass
        
        
        #Split into train, test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.3, shuffle=True)

        #Fit scaler to x train data
        scaler_x = MinMaxScaler()
        scaler_x.fit(self.train_x[self.numeric_column_names])
        self.scaler_x = scaler_x

        self.train_x = scale_and_encode(self.train_x, scaler_x, encoder_x, self.category_column_names, self.numeric_column_names)
        self.test_x = scale_and_encode(self.test_x, scaler_x, encoder_x, self.category_column_names, self.numeric_column_names)

    def train(self, layers=2, nodes=12, epochs=100, verb=2):
        if self.target_is_category:
            self.model = compile_and_fit_classification_model(self.train_x, self.train_y, layers=layers, nodes=nodes, epochs=epochs, verb=verb)
        else:
            self.model = compile_and_fit_regression_model(self.train_x, self.train_y, layers=layers, nodes=nodes, epochs=epochs, verb=verb)
      
        self.model.save("my_mod")     

    def test(self):
        self.eval = self.model.evaluate(self.test_x, self.test_y)
        if self.target_is_category:
            pass
        else:
            self.model.evaluate(self.test_x, self.test_y)
            


    def predict_dict(self, dic):
        new = pd.DataFrame({0:dic}).transpose()
        new = scale_and_encode(new, self.scaler_x, self.encoder_x, self.category_column_names, self.numeric_column_names)
        return self.model.predict(new)