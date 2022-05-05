from flask import Flask, render_template, request  # 导入render_template模块
import pandas as pd
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
        predictionWord = "Your comment is [" + nm_string + "]."

        # make some rules:
        # 全数字，长度为0，输入的为空格，纯字母
        if nm.isdigit()==True or len(nm)==0 or nm.isspace()==True or nm.isalpha():
            probability_float = 10

        if prediction == 'OR' and probability_float > 0.7:
            probability_msg = "http://smartbuyers.ml/wp-content/uploads/2022/05/smiling.png"
            user_tips = "After our system determining, this comment is a fact comment and you can trust it."

        elif prediction == 'OR' and 0.7 >= probability_float:
            probability_msg = "http://smartbuyers.ml/wp-content/uploads/2022/05/thinking.png"
            user_tips = "After our system determining, this comment may be a fact one but can't fully trust it."

        elif prediction == 'CG':
            probability_msg = "http://smartbuyers.ml/wp-content/uploads/2022/05/angry.png"
            user_tips = "After our system determining, this comment is a fake comment."

        else:
            probability_msg = " "
            user_tips = " "

        if probability_float == 10:
            predictionWord = "It may be a invalid review, Please give me a review again."
            probability_msg = "http://smartbuyers.ml/wp-content/uploads/2022/05/thinking.png"
            user_tips = ""

        return render_template("result.html", data=predictionWord, data1=probability_msg, data2=user_tips)  # 调用render_template函数，传入html文件参数


if __name__=="__main__":
    #application.run(port=5000,host="127.0.0.1",debug=True)
    application.debug = True
    application.run()