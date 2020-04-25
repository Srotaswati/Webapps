from flask import Flask, render_template, url_for, request, escape, session

from vsearch import search4letters
from checker import check_logged_in

app=Flask(__name__)

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

def log_request(req: 'flask request',res:str)->None:
    with open('vsearch.log','a') as log:
        print(req.form, req.remote_addr, req.user_agent, res, file=log, sep='|')
        
@app.route('/search4', methods=['POST'])
def do_search()->'html':
    phrase=request.form['phrase']
    letters=request.form['letters']
    title='Here are your results'
    results=str(search4letters(phrase, letters))
    log_request(request, results)
    return render_template('results.html', the_phrase=phrase, the_letters=letters, the_title=title, the_results=results,)

@app.route('/entry')
def entry_page()->'html':
    return render_template('entry.html', the_title='Welcome to search4letters on the web!')

@app.route('/viewlog')
@check_logged_in
def view_the_log()->'html':
    contents=[]
    with open('vsearch.log') as log:
        for line in log:
            contents.append([])
            for item in line.split('|'):
                contents[-1].append(escape(item))
    titles=('Form Data', 'Remote_addr', 'User_agent', 'Results')
    return render_template('viewlog.html', the_title='View log', the_row_titles=titles, the_data=contents,)

app.secret_key = 'MySecretKey'

if __name__=='__main__':
    app.run(debug=True)
