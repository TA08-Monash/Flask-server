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

application = Flask(__name__)


# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')

#url = 'https://www.amazon.com.au/Things-Wanted-Say-but-never/dp/B09VFS57HK/ref=pd_rhf_gw_s_pd_crcd_sccl_1_4/357-9763785-0087255?pd_rd_w=NvlyX&pf_rd_p=3773418f-973d-4eab-a18e-142337dd5ace&pf_rd_r=F67H7EF59TQ8634D0Y2M&pd_rd_r=623e4c6a-2e23-4eb7-becd-7d9e625104a2&pd_rd_wg=4t9Hj&pd_rd_i=B09VFS57HK&psc=1'
#url1 = 'https://www.amazon.com/Long-Walk-Water-Based-Story/dp/0547577311/?_encoding=UTF8&pd_rd_w=NeCTd&pf_rd_p=ba25a0fb-eeb9-4762-9c76-8ca869df5234&pf_rd_r=DW0MXM6SKH8WDYAAS2SZ&pd_rd_r=c9014073-37d7-4d49-869c-e6dacc633794&pd_rd_wg=O9SiM&ref_=pd_gw_exports_top_sellers_unrec'

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

# ua USA
# headers1 = {
#         'authority': 'www.amazon.com',
#         'pragma': 'no-cache',
#         'cache-control': 'no-cache',
#         'dnt': '1',
#         'upgrade-insecure-requests': '1',
#         'User-Agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
#         'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
#         'sec-fetch-site': 'none',
#         'sec-fetch-mode': 'navigate',
#         'sec-fetch-dest': 'document',
#         'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
#     }

def scrape_Url(input = str,reviewnum= int):
    '''
    :param input: user input
    :param reviewnum: reviewnum is the num of review: reviewnum = 1 means only scrape first page's review
    :return:reviews,product_title,product_pic_link
    '''
    global list_comment, list_rating, product_title, product_pic_link
    pattern = re.compile('(?<=www.amazon.).*?(?=/)')
    middleUrl = re.findall(pattern, input)[0]
    # if middleUrl == 'com':
    #     response = requests.get(url=url, headers=headers)
    # else:
    #     response = requests.get(url=url, headers=headers)
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

    # product_title and product_pic_link
    product_title = data['product_title']
    try:
        product_pic_link = soup.find(name='img', attrs={'class':'a-dynamic-image image-stretch-vertical frontImage',"id":"imgBlkFront"}).attrs['src']
    except:
        product_pic_link = ""

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

    #print(list_comment)
    #print(list_rating)
    #global list_comment, list_rating, product_title, product_pic_link
    return list_comment, product_title, product_pic_link

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
        scrape_Url(url, 2)
        #nm_list = [nm_string]

        #model
        df = pd.read_csv('fake reviews dataset.csv')
        text_clf_svm = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf-svm', CalibratedClassifierCV(SGDClassifier(loss='hinge', penalty='l2',
                                                                                  alpha=1e-3, random_state=42)))])
        text_clf_svm.fit(df['text_'], df['label'])
        ##############

        ##############
        list_fact = []
        list_fact_pbbt = []
        list_fake = []
        list_fake_pbbt = []
        list_total_all = 0
        n = 0
        i = 0
        rating_total = 0

        for comment in list_comment:
            list_running = [comment]
            #list_total_all = list_total_all + list_rating[i]
            prediction = text_clf_svm.predict(list_running)
            probability = np.amax(text_clf_svm.predict_proba(list_running), axis=1)
            probability_float = float(probability)

            if prediction == 'OR':
                list_fact.append(comment)           #前端调用的真实评论list
                list_fact_pbbt.append(probability_float)
                #rating_total = rating_total + list_rating[n]
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

    # check the len of two lists
    #if len(sorted_fact) < 5:
    #    for nums in range(len(sorted_fact), 5):
    #        sorted_fact.append([" ", " "])

    #if len(sorted_fake) < 5:
    #    for nums in range(len(sorted_fake), 5):
    #        sorted_fake.append([" ", " "])

    # try fact
    try:
        fact1 = sorted_fact[0][0]
        fact1_pbbt = "The probability: " + str(round(sorted_fact[0][1] * 100, 1)) + "%"
    except:
        fact1 = " "
        fact1_pbbt = " "

    try:
        fact2 = sorted_fact[1][0]
        fact2_pbbt = "The probability: " + str(round(sorted_fact[1][1] * 100, 1)) + "%"
    except:
        fact2 = " "
        fact2_pbbt = " "

    try:
        fact3 = sorted_fact[2][0]
        fact3_pbbt = "The probability: " + str(round(sorted_fact[2][1] * 100, 1)) + "%"
    except:
        fact3 = " "
        fact3_pbbt = " "

    try:
        fact4 = sorted_fact[3][0]
        fact4_pbbt = "The probability: " + str(round(sorted_fact[3][1] * 100, 1)) + "%"
    except:
        fact4 = " "
        fact4_pbbt = " "

    try:
        fact5 = sorted_fact[4][0]
        fact5_pbbt = "The probability: " + str(round(sorted_fact[4][1] * 100, 1)) + "%"
    except:
        fact5 = " "
        fact5_pbbt = " "

    # Try fake
    try:
        fake1 = sorted_fake[0][0]
        fake1_pbbt = "The probability: " + str(round(sorted_fake[0][1] * 100, 1)) + "%"
    except:
        fake1 = " "
        fake1_pbbt = " "

    try:
        fake2 = sorted_fake[1][0]
        fake2_pbbt = "The probability: " + str(round(sorted_fake[1][1] * 100, 1)) + "%"
    except:
        fake2 = " "
        fake2_pbbt = " "

    try:
        fake3= sorted_fake[2][0]
        fake3_pbbt = "The probability: " + str(round(sorted_fake[2][1] * 100, 1)) + "%"
    except:
        fake3 = " "
        fake3_pbbt = " "

    try:
        fake4 = sorted_fake[3][0]
        fake4_pbbt = "The probability: " + str(round(sorted_fake[3][1] * 100, 1)) + "%"
    except:
        fake4 = " "
        fake4_pbbt = " "

    try:
        fake5 = sorted_fake[4][0]
        fake5_pbbt = "The probability: " + str(round(sorted_fake[4][1] * 100, 1)) + "%"
    except:
        fake5 = " "
        fake5_pbbt = " "

##############
    return render_template("resultUrl.html",
                            data_head=product_title,
                            data_picture=product_pic_link,
                            data_link=url,
                            data_amount_rating=i,
                            data_adj_amount_rating=n,
                            data_fact1=fact1,
                            data_fact2=fact2,
                            data_fact3=fact3,
                            data_fact4=fact4,
                            data_fact5=fact5,
                            data_fact1_pbbt=fact1_pbbt,
                            data_fact2_pbbt=fact2_pbbt,
                            data_fact3_pbbt=fact3_pbbt,
                            data_fact4_pbbt=fact4_pbbt,
                            data_fact5_pbbt=fact5_pbbt,
                            data_fake1=fake1,
                            data_fake2=fake2,
                            data_fake3=fake3,
                            data_fake4=fake4,
                            data_fake5=fake5,
                            data_fake1_pbbt=fake1_pbbt,
                            data_fake2_pbbt=fake2_pbbt,
                            data_fake3_pbbt=fake3_pbbt,
                            data_fake4_pbbt=fake4_pbbt,
                            data_fake5_pbbt=fake5_pbbt,
                            rating_100=round(n/i * 100, 1),
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