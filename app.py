import torch
from flask import Flask,render_template,request


import Predict

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Predict.ANN()
model.load_state_dict(torch.load('runs/model_weights.pth',map_location=torch.device(device)))
model.to(device)
model.eval()
model.double()


def prediction(inputname):
    X = Predict.TwitAPI(username=inputname)

    X = X.values
    X_pred = torch.tensor(X,device=device)

    y_pred_list1 = []
    with torch.no_grad():
        for i in range(len( X_pred)):
            y_train_pred = model(X_pred[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list1.append(y_pred_tag.cpu().numpy())
    prediiction = int(y_pred_list1[0])
    if prediiction == 0:
        return ("Prediction for " + inputname + " is Bot")
    else:
        return("Prediction for " + inputname + " is Human")



@app.route('/form')
def form():
    form_data = {'key1(field1_name)': 'value1(field1_value)'}
    return render_template('form.html')
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        #form_data = request.form
        form_data = {'User': str(prediction(str(list(request.form.values())[0])))}
        print(form_data)
        return render_template('data.html',form_data = form_data)


app.run(host='0.0.0.0', port=5000)