import json

class Tree:
    def __init__(self):
        self.tree = dict()
        self.choices = list()
    
    def load(self, file):
        self.tree = json.loads(open(file).read())
        self.extrac_choices(self.tree)

    def extrac_choices(self, node):
        if isinstance(node, dict):
            self.choices.append(node["choice"])
            node["choice"] = len(self.choices)-1
            self.extrac_choices(node["right"])
            self.extrac_choices(node["left"])
        return

    def tree_to_html(self):
        return self.step_to_html(self.tree)

    def step_to_html(self, node):
        ul = '<ul class="keep">';
        if isinstance(node, dict):
            ul += '<li class="stem">' 
            ul +=  node["index"] 
            ul += " 划分点=" 
            ul += str(node["value"]) 
            ul += " "
            ul += createBtn(node["choice"])
            ul += "</li>"
            ul += '<ul class="children">'
            ul += "<li> < " + str(node["value"]) + "</li>"
            ul += self.step_to_html(node["left"])
            ul += "<li> >=" + str(node["value"]) + "</li>"
            ul += self.step_to_html(node["right"])
            ul += "</ul>"
        else:
            if node == 0:
                ul += '<li class="rumor_leaf">' + "谣言" + "</li>"
            else:
                ul += '<li class="news_leaf">' + "新闻" + "</li>"
        ul += "</ul>"
        return ul

def createBtn(index: int):
    btn = '<a href="/choice?page_no=0&page_cnt=20&choiceId=' + str(index) + '">choices</a>'
    return btn