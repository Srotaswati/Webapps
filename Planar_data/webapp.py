from flask import Flask, render_template, url_for, request, redirect, session, flash

from forms import ContactForm
from getdata import *
from planardata import *
from DBcm import UseDatabase
from checker import check_logged_in

app=Flask(__name__)
app.config['dbconfig'] = {'host':'127.0.0.1', 'user':'bot', 'password':'simple', 'database':'deeplearning'} 

@app.route('/entry', methods = ['GET', 'POST'])
def contact():
   form = ContactForm()
   
   if request.method == 'POST':
      if form.validate() == False:
         flash('All fields are required.')
         return render_template('entry.html', form = form)
      else:
         return 'Details entered successfully'
   elif request.method == 'GET':
         return render_template('entry.html', form = form)

@app.route('/login')
def do_login()-> str:
    session['logged_in'] = True
    return 'You are now logged in'

@app.route('/logout')
def do_logout() -> str:
    session.pop('logged_in')
    session.clear()
    return redirect(url_for('index'))

@app.route('/')
@app.route('/index')
def index()->'html':
    return render_template('index.html')

def log_request(req:'flask request', alpha:float, reg_param:float, dropout:float)->None:
    with UseDatabase(app.config['dbconfig']) as cursor:
        _SQL = """insert into log (dataset, optimizer, ip, browser_string, alpha, reg_param, dropout) values(%s,%s,%s,%s,%f,%f,%f)"""
        cursor.execute(_SQL,(req.form['dataset'], req.form['optimizer'], req.remote_addr, req.user_agent.browser, alpha, reg_param, dropout,))
    
        
@app.route('/results', methods=['POST'])
def do_search()->'html':
    dataset=request.form['dataset']
    opt=request.form['optimizer']
    title='Here are your results'
    train_x, train_y = get_training_data(dataset)
    layers_dims = [train_x.shape[0], 5, 2, 1]
    model = {}
    with timer() as elapsed:
       model = neural_network(train_x, train_y, layers_dims, optimizer=opt, num_epochs = 10000, lambd=0.7, print_cost = False)
    plt.show()
    log_request(request, model['alpha'], model['reg_param'], model['keep_prob'])
    return render_template('results.html', the_dataset=dataset, the_optimizer=optimizer, the_title=title, the_results=model['alpha'])

@app.route('/entry')
def entry_page()->'html':
    return render_template('entry.html', the_title='Welcome to planar data on the web!')

@app.route('/viewlog')
@check_logged_in
def view_the_log()->'html':
    with UseDatabase(app.config['dbconfig']) as cursor:
        _SQL = """select dataset, optimizer, ip, browser_string, alpha, reg_param, dropout from log"""
        cursor.execute(_SQL)
        contents = cursor.fetchall()
    titles=('Dataset', 'Optimizer', 'Remote_addr', 'User_agent', 'Learning Rate', 'Regularization', 'Dropout Rate')
    return render_template('viewlog.html', the_title='View log', the_row_titles=titles, the_data=contents,)

app.secret_key = 'MySecretKey'

if __name__=='__main__':
    app.run(debug=True)
