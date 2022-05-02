from flask import Flask, render_template, request  # 导入render_template模块
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV


application = Flask(__name__)

@application.route('/')
def index():
    msg = 'Insert the review/comment you want to verify (if it is fake or real)'
    return render_template("index.html",data =msg)   #调用render_template函数，传入html文件参数

@application.route('/loginProcess',methods=['POST','GET'])
def loginProcesspage():
    if request.method=='POST':
        nm = request.form['nm']    #获取姓名文本框的输入值
        nm_string = str(nm)
        nm_list = [nm_string]
        df = pd.read_csv('fake reviews dataset.csv')
        text_clf_svm = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf-svm', CalibratedClassifierCV(SGDClassifier(loss='hinge', penalty='l2',
                                                                                  alpha=1e-3, random_state=42)))])
        text_clf_svm.fit(df['text_'], df['label'])
        prediction = text_clf_svm.predict(nm_list)
        probability = np.amax(text_clf_svm.predict_proba(nm_list), axis=1)
        probability_float = float(probability)

        if probability_float > 0.9:
            probability_msg = " and the probability > 90%"
        elif probability_float > 0.8:
            probability_msg = " and the probability > 80%"
        elif probability_float > 0.7:
            probability_msg = " and the probability > 70%"
        elif probability_float > 0.6:
            probability_msg = " and the probability > 60%"
        elif probability_float > 0.5:
            probability_msg = " and the probability > 50%"
        else:
            probability_msg = " and the probability < 50%"

        if prediction == 'OR':
            predictionWord = "[" + nm_string + '] may be a Fact comment, '
        else:
            predictionWord = "[" + nm_string + '] may be a Computer Generated comment, '
        return render_template("result.html", data=predictionWord, data1=probability_msg)  # 调用render_template函数，传入html文件参数


if __name__=="__main__":
    #application.run(port=5000,host="127.0.0.1",debug=True)
    application.debug = True
    application.run()