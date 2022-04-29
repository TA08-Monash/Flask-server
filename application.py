from flask import Flask, render_template, request  # 导入render_template模块

application=Flask(__name__)

@application.route('/')
def index():
    msg = 'Please input the content which you want to identify.'
    return render_template("index.html",data =msg)   #调用render_template函数，传入html文件参数

@application.route('/loginProcess',methods=['POST','GET'])
def loginProcesspage():
    if request.method=='POST':
        nm = request.form['nm']     #获取姓名文本框的输入值
        return render_template("result.html", data=nm)  # 调用render_template函数，传入html文件参数


if __name__=="__main__":
    application.run(port=5000,host="127.0.0.1",debug=True)