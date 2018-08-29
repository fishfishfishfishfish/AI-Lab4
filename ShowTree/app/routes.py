from flask import render_template
from app import app
from app import load_tree
from flask import request
import glob

tree = load_tree.Tree()
main_dir = "E://Documents//Python_Project//AI-Lab4//ShowTree"
tree_dir = main_dir + "//app//json//" 
# tree.load("E://Documents//Python_Project//AI-Lab4//ShowTree//app//C45_decision_tree.json")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    file_list = [file_name.split('\\')[1] for file_name in glob.glob(tree_dir + '*')]
    return render_template('gen_file.html', file_list=file_list)

@app.route('/graph', methods=['GET', 'POST'])
def graph():    
    tree_name = request.form['name']
    # tree = load_tree.Tree()
    tree.load(tree_dir + tree_name)
    tree_str = tree.tree_to_html()
    return render_template('gen_tree.html', tree_str=tree_str)


@app.route('/choice', methods=['GET', 'POST'])
def choice():
    page_no = int(request.args.get('page_no'))
    page_cnt = int(request.args.get('page_cnt'))
    choiceId = int(request.args.get('choiceId'))
    choices = tree.choices[choiceId][page_no:page_no+page_cnt]
    print(page_no, page_no+page_cnt, len(tree.choices[choiceId]))
    return render_template('gen_choice.html', choices=choices, page_no=page_no, page_cnt=page_cnt, choiceId=choiceId, choices_num=len(tree.choices[choiceId]))