

from flask import Flask,render_template, request, url_for, flash, redirect
from anime_result import get_suggestions
import numpy as np
#print(anime_df.tail())

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('result.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        raw_input= [x for x in request.form.values()]
        anime_query = str(raw_input[0])

        
        if request.form['submit_button'] == 'get_recommendation':
            result, flag = get_suggestions(anime_query)
            if flag ==0:
                result= np.array(result)
                query_anime= result[0].tolist()
                suggestion_anime= result[1:, :].tolist()
                print(suggestion_anime[0])
           
                #return render_template('result.html', tables=[result.to_html(classes = 'my_class" id = "my_id',header="true", index=False)], flag= flag)
                return render_template('result.html', query_anime= query_anime,suggestion_anime=suggestion_anime, flag= flag)
            elif flag == 1:
                print(result)
                return render_template('result.html', tables=[result.to_html(classes = 'my_class" id = "my_id',header="true", index=False)], flag= flag)
                #return render_template('result.html', tables=result, flag= flag)
       
       
        
           
       
       
if __name__ == '__main__':
    #import webbrowser
    #webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug= True)
