from flask import Flask, render_template, request  # 导入render_template模块
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from selectorlib import Extractor
import re
import requests
from bs4 import BeautifulSoup
import pickle
import random

application = Flask(__name__)


# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')

# scrape reviews
def scrape(url):
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create
    return e.extract(r.text)

# ua AU
headers = {
    'authority': 'www.amazon.com',
    'pragma': 'no-cache',
    'cache-control': 'no-cache',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-dest': 'document',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
}

user_agent = ['Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1 ',
             'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0 ',
             'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50 ',
             'Opera/9.80 (Windows NT 6.1; U; zh-cn) Presto/2.9.168 Version/11.50 ',
             'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 2.0.50727; SLCC2; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.3; .NET4.0C; Tablet PC 2.0; .NET4.0E) ',
             'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; InfoPath.3) ',
             'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; GTB7.0) ',
             'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1) ',
             'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ',
             'Mozilla/5.0 (Windows; U; Windows NT 6.1; ) AppleWebKit/534.12 (KHTML, like Gecko) Maxthon/3.0 Safari/534.12 ',
             'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.3; .NET4.0C; .NET4.0E) ',
             'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.3; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0) ',
             'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.3 (KHTML, like Gecko) Chrome/6.0.472.33 Safari/534.3 SE 2.X MetaSr 1.0 ',
             'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.3; .NET4.0C; .NET4.0E) ',
             'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.41 Safari/535.1 QQBrowser/6.9.11079.201 ',
             'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.3; .NET4.0C; .NET4.0E) QQBrowser/6.9.11079.201 ',
             'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0) ']

def scrape_Url(input = str,reviewnum= int):
    '''
    :param input: user input
    :param reviewnum: reviewnum is the num of review: reviewnum = 1 means only scrape first page's review
    :return:reviews,product_title,product_pic_link
    '''
    global list_comment, list_rating, product_title
    pattern = re.compile('(?<=www.amazon.).*?(?=/)')
    middleUrl = re.findall(pattern, input)[0]

    num = random.randint(0, 17)
    headers['user-agent'] = user_agent[num]
    response = requests.get(url=input, headers=headers)

    # load webpage
    page_text = response.text
    soup = BeautifulSoup(page_text, 'html.parser')  #
    test = soup.find(name='a', attrs={'data-hook': 'see-all-reviews-link-foot', 'class': 'a-link-emphasis a-text-bold'})
    package = soup.find(name='a', attrs={'data-hook': 'see-all-reviews-link-foot', 'class': 'a-link-emphasis a-text-bold'}).attrs[
        'href']

    # see all reviews link
    baseUrl = 'https://www.amazon.'
    finalurl = baseUrl + middleUrl + package

    data = scrape(finalurl)
    reviews = []

    # product_title
    product_title = data['product_title']
    reviews = data['reviews']
    i = 0
    while (data['next_page'] != None and i < reviewnum - 1):
        finalurl = baseUrl + middleUrl + data['next_page']
        data = scrape(finalurl)
        temp = data['reviews']
        result = len(temp)
        if result != 0:
            reviews = reviews + temp
        i = i + 1

    list_comment = []

    for review in reviews:
        list_comment.append(review["content"])

    return list_comment, product_title

@application.route('/')
def index():
    msg = 'Insert the review/comment you want to verify (if it is fake or real)'
    return render_template("index.html",data =msg)   #调用render_template函数，传入html文件参数

@application.route('/checkUrl')
def t():
    return render_template("checkUrl.html")

@application.route('/urlProcess',methods=['POST','GET'])
def urlProcesspage():
    if request.method=='POST':
        nm = request.form['nm']
        url = str(nm)    #获取姓名文本框的输入值
        try:
            scrape_Url(url, 2)
        except:
            return render_template("Error – SmartBuyers.html")

        #model
        with open('static/SGD.pickle', 'rb') as handle:
            text_clf_svm = pickle.load(handle)

        ##############
        list_fact = []
        list_fact_pbbt = []
        list_fake = []
        list_fake_pbbt = []
        n = 0
        i = 0

        for comment in list_comment:
            list_running = [comment]
            prediction = text_clf_svm.predict(list_running)
            probability = np.amax(text_clf_svm.predict_proba(list_running), axis=1)
            probability_float = float(probability)

            if prediction == 'OR':
                list_fact.append(comment)           #前端调用的真实评论list
                list_fact_pbbt.append(probability_float)
                n = n + 1
            if prediction == 'CG':
                list_fake.append(comment)
                list_fake_pbbt.append(1-probability_float)
            i = i + 1

    df_fact = pd.DataFrame({"fact_reviews":list_fact, "fact_pbbt":list_fact_pbbt})
    df_fake = pd.DataFrame({"fake_reviews": list_fake, "fake_pbbt": list_fake_pbbt})

    # sort pbbt in a descending order
    df_fact.sort_values(by=['fact_pbbt'], inplace=True, ascending=False)
    df_fake.sort_values(by=['fake_pbbt'], inplace=True)

    sorted_fact = df_fact.values.tolist()
    sorted_fake = df_fake.values.tolist()

    fact_list_render = []
    if len(sorted_fact) > 4:
        for num in range(0, 5):
            fact_list_render.append([sorted_fact[num][0],"Probability: " + str(round(sorted_fact[num][1] * 100, 1)) + "% "])
    else:
        for num in range(0,len(sorted_fact)):
            fact_list_render.append([sorted_fact[num][0], "Probability: " + str(round(sorted_fact[num][1] * 100, 1)) + "% "])
        for num in range(len(sorted_fact),5):
            fact_list_render.append([" ", " "])

    fake_list_render = []
    if len(sorted_fake) > 4:
        for num in range(0, 5):
            fake_list_render.append([sorted_fake[num][0],"Probability " + str(round(sorted_fake[num][1] * 100, 1)) + "% "])

    else:
        for num in range(0,len(sorted_fake)):
            fake_list_render.append([sorted_fake[num][0], "Probability: " + str(round(sorted_fake[num][1] * 100, 1)) + "% ","static/Chat.svg"])

        for num in range(len(sorted_fake),5):
            fake_list_render.append([" ", " ","http://smartbuyers.ml/wp-content/uploads/2022/05/nomore.png"])
##############

    #return render_template("template.html",
    return render_template("resultUrl.html",
                            data_head=product_title,
                            data_link=url,
                            data_amount_rating=i,
                            data_adj_amount_rating=n,
                            data_fake_amount=i-n,
                            data_fact1=fact_list_render[0][0],
                            data_fact2=fact_list_render[1][0],
                            data_fact3=fact_list_render[2][0],
                            data_fact4=fact_list_render[3][0],
                            data_fact5=fact_list_render[4][0],
                            data_fact1_pbbt=fact_list_render[0][1],
                            data_fact2_pbbt=fact_list_render[1][1],
                            data_fact3_pbbt=fact_list_render[2][1],
                            data_fact4_pbbt=fact_list_render[3][1],
                            data_fact5_pbbt=fact_list_render[4][1],
                            data_fake1=fake_list_render[0][0],
                            data_fake2=fake_list_render[1][0],
                            data_fake3=fake_list_render[2][0],
                            data_fake4=fake_list_render[3][0],
                            data_fake5=fake_list_render[4][0],
                            data_fake1_pbbt=fake_list_render[0][1],
                            data_fake2_pbbt=fake_list_render[1][1],
                            data_fake3_pbbt=fake_list_render[2][1],
                            data_fake4_pbbt=fake_list_render[3][1],
                            data_fake5_pbbt=fake_list_render[4][1],
                            rating_100=str(round(n/i * 100, 0))+"%",
                            comment_pics="static/Chat.svg",
                            comment_pic1=fake_list_render[0][2],
                            comment_pic2=fake_list_render[1][2],
                            comment_pic3=fake_list_render[2][2],
                            comment_pic4=fake_list_render[3][2],
                            comment_pic5=fake_list_render[4][2],
                            rating_percentage="c100 p"+str(int(round(n/i * 100, 1))),
                            rating_623=(round(n/i, 1) * 623))  # 调用render_template函数，传入html文件参数


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
        #predictionWord = "Your comment is [" + nm_string + "]."
        predictionWord = " "

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