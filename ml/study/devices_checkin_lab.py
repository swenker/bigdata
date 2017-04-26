from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


"""
mysql> select * from devices_counter_byday into outfile'/tmp/devices_counter_byday.csv' fields terminated by ',';

"""
DATA_DIR='/home/wenjusun/bigdata/data/checkin_data'


def date_str_toordinal(date_str):
    x_date = datetime.strptime(date_str, '%Y-%m-%d')
    return datetime.toordinal(x_date)

def load_device_counter_byday():
    filename=DATA_DIR+'/devices_counter_byday.csv'
    X=[]
    X_raw=[]
    Y_unique=[]
    Y_all=[]
    with open(filename) as f:
        # print "lines read: %d " % len(f.)
        for line in f:
            fields = line.split(',')
            x_date = fields[0]
            y_unique = int(fields[1])
            y_all = int(fields[2])

            X_raw.append(x_date)
            X.append([date_str_toordinal(x_date)])

            Y_unique.append(y_unique)
            Y_all.append(y_all)
    return X,Y_unique,Y_all,X_raw



def predict_device_byday_linear_regression():
    X,Y_unique,Y_all,X_raw = load_device_counter_byday()
    # print X
    # print Y_unique
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    training_size = 160
    # model.fit(X[:training_size],Y_unique[:training_size])
    model.fit(X[:training_size],Y_all[:training_size])

    start_index = 180
    end_index = 190
    X_to_predict = X[start_index:end_index]
    # X_to_predict.append([date_str_toordinal('2017-04-18')])
    # X_to_predict.append([date_str_toordinal('2017-03-27')])

    print X_to_predict
    # Y_real = Y_unique[start_index:end_index]
    Y_real = Y_all[start_index:end_index]
    print X_raw[start_index:end_index]
    y_predicted=model.predict(X_to_predict)
    # print y_predicted
    y_predicted = np.array(y_predicted).astype(int)
    print y_predicted
    print Y_real
    # print y_predicted - np.array(Y_real)

    # plt.subplot(111)
    # plt.scatter(X_to_predict,Y_real,c='r')
    plt.scatter(X_to_predict,y_predicted)
    # plt.plot(X_to_predict,y_predicted)
    plt.show()

def predict_device_byday_SVR():
    X,Y_unique,Y_all,X_raw = load_device_counter_byday()

    from sklearn.svm import SVR
    model = SVR()
    # model = SVR(kernel='linear')
    training_size = 160
    # model.fit(X[:training_size],Y_unique[:training_size])
    model.fit(X[:training_size],Y_all[:training_size])

    start_index = 180
    end_index = 190
    X_to_predict = X[start_index:end_index]
    # X_to_predict.append([date_str_toordinal('2017-04-18')])
    # X_to_predict.append([date_str_toordinal('2017-03-27')])

    print X_to_predict
    # Y_real = Y_unique[start_index:end_index]
    Y_real = Y_all[start_index:end_index]
    print X_raw[start_index:end_index]
    y_predicted=model.predict(X_to_predict)
    # print y_predicted
    y_predicted = np.array(y_predicted).astype(int)
    print y_predicted
    print Y_real
    # print y_predicted - np.array(Y_real)

    # plt.subplot(111)
    # plt.scatter(X_to_predict,Y_real,c='r')
    plt.scatter(X_to_predict,y_predicted)
    # plt.plot(X_to_predict,y_predicted)
    plt.show()



if __name__ == '__main__':

    # predict_device_byday( )
    predict_device_byday_SVR( )

