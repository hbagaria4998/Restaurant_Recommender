import flask
import pickle
import pandas as pd
import Rec_fx as rf   

# Use pickle to load in the pre-trained model and preprocessed parts
with open('rr_model.pkl', 'rb') as rr:
    model1 = pickle.load(rr)
with open('list1.pkl', 'rb') as ll:
    list1 = pickle.load(ll)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates',)
app.config['DEBUG'] = True
# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Extract the input
        i = flask.request.form['user']
        

        kn_ps,rc = rf.sample_train_recommendation(model1,list1[0],list1[2],[i],5,'name',mapping=list1[3].mapping()[2],tag='category',
                              user_features = list1[4],item_features=list1[5])
       


        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',kp=kn_ps,r = rc,i = i)


if __name__ == '__main__':
    app.run()
