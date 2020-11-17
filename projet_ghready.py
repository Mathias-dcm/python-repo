import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

#pip install cufflinks --upgrade
import cufflinks as cf
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score 
import plotly.figure_factory as ff
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from time import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#helooooo

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}





# Dessiner layout 




app.layout = html.Div([
    
    html.Div([ 
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px', 
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    
    html.Div(id='output-data-upload',
             style={ 'width': '10%'}
             ),    
    
        
    
    dcc.Dropdown(
        id='cible',
        children=html.Div([
            'Choisir variable cible',
        ]),
         placeholder="Choix de la variable à prédire", 
        options=[
            {'label':'data', 'value': 'data'},
        ],
        multi=False,
        style={'backgroundColor': '#5E3E3E'},
        className='stockselector',
       #value=[]
        ),
    
    
    
    
       dcc.Dropdown(
        id='pre_algo',
        children=html.Div([
            'Choisir variable cible',
        ]),
        placeholder="Choix le type",
        options=[
            {'label':'data', 'value': 'data'},
        ],
        multi=False,
        style={'backgroundColor': '#5E3E3E'},
        className='stockselector',
        value=[]
        ),
    
    
    
    
    dcc.Dropdown(
        id='algo',
        children=html.Div([
            'Choisir variable cible',
        ]),
        placeholder="Choix de l'algorithme",
        options=[
            {'label':'data', 'value': 'data'},
        ],
        multi=True,
        style={'backgroundColor': '#5E3E3E'},
        className='stockselector',
       #value=[]
        ),
    
    
    
    
    
    
    
    
    
    dcc.Dropdown(
        id='predire',
        children=html.Div([
            'Choisir variable cible',
        ]),
        placeholder="Choix des variables prédictives", 
         style={'backgroundColor': '#5E3E3E'},
        options=[
            
          
        ],
        multi=True,
    
        className='stockselector',
#        value=['data']
        ),
    
    html.Div(id='tabletype'),
    html.Div(id='data'),
    html.Div(id='graph_PCA',),

    
   
   
    ], style={'width': '30%', 'display': 'inline-block'}) ,
    
    
    
   html.Div([ 
       
       
       #Création d'onglets pour afficher les résultats des différentes méthodes 
       html.Div(id='div_onglets', className='control-tabs', children=[
            dcc.Tabs(id='tabs_onglets', value='tabs', children=[
            dcc.Tab(id='tab1', value='tab-1'),
            dcc.Tab(id="tab2", value='tab-2'),
            dcc.Tab(id="tab3", value='tab-3'),
            ]),
            #On inclut tous les éléments graphiques dont on a besoin 
            html.Div(id='onglets_content',children=[
                
                html.Div(id='acc'),
                html.Div(id='dtr_continue'),
                html.Div(id='dtc_continue'),
                html.Div(id='neuron'),
                html.Div(id='gradient_class'),
                html.Div(id='adl'),
                html.Div(id='ensemble'),
              
                  html.Div(id='graph',),
                  html.Div(id='graph_dtc',),
                  html.Div(id='graph1',),
                  html.Div(id='graph2',),
                  html.Div(id='graph_adl'),
                  html.Div(id='reglog')
                ])
        
            ]),
    
    
    ],style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
    
  

],
   )


#Affichage du contenu spécifique à chaque onglet 
@app.callback(Output('onglets_content', 'children'),
              [Input('tabs_onglets', 'value')])

def update_content_ongglets(tab):
    
    if tab == 'tab-1':
        return html.Div([
            html.Div(id='acc'),
            html.Div(id='graph_dtc',),
            html.Div(id='graph1',),
            html.Div(id='reglog'),
           
    
        ])
    elif tab=="tab-2":
        return html.Div([
            html.Div(id='dtr_continue'),
            html.Div(id='dtc_continue'),
            html.Div(id='graph',)
            ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div(id='graph_adl'),
             html.Div(id='neuron'),
             html.Div(id='graph2',)
        ])


#Modification du label de chaque onglet en fonction du type de la variable cible 
@app.callback(Output('tab1', 'label'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_label_tab1(cible,contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if cible:
            type_var=QT_function0(df,cible)
            if "Qualitative" in type_var:
                return "Régression logistique" 
            else :
                return "Régression" #D T classifier #adl   #reg  SGB  DT regressor 
    return "Méthode 1"

@app.callback(Output('tab2', 'label'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_label_tab2(cible,contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if cible:
            return "Arbre de décision" 
    return "Méthode 2"

@app.callback(Output('tab3', 'label'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_label_tab3(cible,contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if cible:
            type_var=QT_function0(df,cible)
            if "Qualitative" in type_var:
                return "Analyse Discriminante Linéaire" 
            else :
                return "SGB" 
    return "Méthode 3"



# Fonction qui détermine les variables qualitatives/quantitatives du CIBLE


def QT_function0(df,value):
    out=[]
    if str(df.dtypes[str(value)])=='object':
        out="Qualitative"
    else:
        if (len(np.unique(df[str(value)]))<6):
            out="Qualitative"
        else:
            out="Quantitative"
    return [str(out)]




#Chargement du fichier 
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
                        ]),
    return df

######################## AFFICHER LA TABLE DE TYPE #####################################
        
"""
@app.callback(Output('tabletype', 'children'),
              [Input('cible', 'value')],[Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def tabletype(value,contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        typelist=list()
        #collist=list(df.columns)
        for col in df:
            typevalue=QT_function0(df,col)
            typelist.append(typevalue)
    #collist=tuple(collist)    
    #typelist=tuple(typelist)
    #listfinal=np.column_stack((collist))
        dftype=pd.DataFrame(typelist,columns=['Nom de Variable','Type de Variable'])
        return html.Div([
                        dash_table.DataTable(
                                id='type',
                                data=dftype.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in dftype.columns]
            )])


"""



@app.callback(Output('pre_algo', 'options'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_output(value, contents,filename): 
    options = []   

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if value:
            out=QT_function0(df, str(value))
            options=[{'label':name, 'value': name}  for name in out]
    return options



################ PRPOSE ALGO POUR CHAQUE VARIABLE ############################


def QT_function(value):
    output=[]
#    if str(df.dtypes[str(value)])=='object':
#        out="Qualitative"
#    else:
#        out="Quantitative"
    if value=="Qualitative":
        output= ["regression logistique", "Decision tree Classifier","Analyse Discriminante linéaire"]
    else:
        output=["Regression", "SGB", "Decision tree Regressor"]
    return output




@app.callback(Output('algo', 'options'),
              [Input('pre_algo', 'value')])

def update_output00(value): 
    options = []   

#    if contents:
#        contents = contents[0]
#        filename = filename[0]
#        df = parse_contents(contents, filename)
    if value:
        out=QT_function(str(value))
        options=[{'label':name, 'value': name}  for name in out]
    return options




# Mise à jour dropdown


@app.callback(Output('cible', 'options'),
              [Input('upload-data', 'contents')], 
              [State('upload-data', 'filename')])

def update_output1(contents, filename):
    options = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        options=[{'label':name, 'value': name} for name in df.columns.tolist()]#+[{'label':'data', 
                                                                                 # 'value':'data'}]
    return options
    



@app.callback(Output('predire', 'options'),
              [Input('upload-data', 'contents'),
              Input('cible','value')], 
              [State('upload-data', 'filename')])

def update_output2(contents, value,filename):
    options = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if value:
            del df[value]
        options=[{'label':name, 'value': name} for name in df.columns.tolist()]
    return options



##### mise à jour les donnes corespond à la variable qu'on a choisi



#@app.callback(Output('data', 'children'),
#              [Input('cible', 'value')], [Input('upload-data', 'contents')],
#              [State('upload-data', 'filename')])

#def update_output3(value,contents,filename):
#    children = html.Div()
#    if contents:
#        contents=contents[0]
#        filename=filename[0]
#        df=parse_contents(contents,filename)
      # children=html.Div([dash_table.DataTable(data=df.to_dict('row'), columns=value)])
#        if str(value)=='data':
#            children=html.Div([dash_table.DataTable(data=df.to_dict('row'),columns=
#                                                    [{'name':i,'id':i}for i in df.columns])])
#        else:
#            children=html.Div([dash_table.DataTable(data=df.to_dict('row'),columns=
#                                             [{'name':str(value),'id':str(value)}])])
#    return children
    





##############################################################################
###########################REGRESSION LINEAIRE #############################
##############################################################################


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold
import numpy as np


# recherche hyperparametre

def lin(df,value,variables):
    c_space = np.logspace(-4, 0, 20)
    params = {'alpha': c_space}
#    df_bis=pd.get_dummies(df)
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    linear=Ridge(normalize=True)
    linear_cv=GridSearchCV(linear, params, cv=5)
    linear_cv.fit(X_train,y_train)
    y_pre=linear_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    score=r2_score(y_test, linear_cv.best_estimator_.predict(X_test))
    return [score, data_frame]
    
    






# Import GradientBoostingRegressor


from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
def gradient(df,value,variables):
    params={'n_estimators':[16,32,64,100,200], 'learning_rate':[0.25,0.1,0.05,0.025],
            'max_depth':[1,2,4,8], 'subsample': [0.5,0.9,1], 'max_features':[0.5,0.75]}
    gb = GradientBoostingRegressor()
 
    df_bis=df.iloc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    gb_cv=GridSearchCV(gb, params, cv=KFold(5), n_jobs=-1)
    gb_cv.fit(X_train,y_train)
    y_pre=gb_cv.best_estimator_.predict(X)
    dict={'classe reel':y, 'classe predict': y_pre}
    data_frame=pd.DataFrame(dict)
    score=r2_score(y_test, gb_cv.best_estimator_.predict(X_test))
    return [score, data_frame]
    
    
    
   











##############################################################################
########################### DECISION TREE ####################################
##############################################################################




# recherche hyperparametre

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
scoring = make_scorer(r2_score)

def dtr_continue(df,value, variables):
    
    params = {"max_depth": [3,6,9,12, None],
              "min_samples_leaf": np.arange(1,9,1),
              "criterion": ["mse", "friedman_mse", "mae"]}
   
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    dt = DecisionTreeRegressor()
    dt_cv=GridSearchCV(dt, params, cv=5, scoring=scoring, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=r2_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    return [acc, data_frame]
    


# DECISION TREE CLASSIFIER


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, accuracy_score
scoring = make_scorer(r2_score)

def dtc_continue(df,value,variables):
    
    params = {"max_depth": [3,6,9,12, None],
              "min_samples_leaf": np.arange(1,9,1),
              "criterion": ["gini", "entropy"]}
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    dt = DecisionTreeClassifier()
    dt_cv=GridSearchCV(dt, params, cv=5, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=accuracy_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
#    dict={'classe réelle':y, 'classe predicte': y_pre}
#    data_frame=pd.DataFrame(dict)
    return [acc]






     
    
    
# Regression/reseau_neuronne/arbre_decision avec le meilleur parametre    
    
# Regression  
    
#@app.callback(Output('accuracy', 'options'),
#              [Input('cible', 'value')], [Input('upload-data', 'contents')],[Input('algo', 'value')],
#              [State('upload-data', 'filename')])

#def update_output4(value1,contents, value2,filename):
#    options = []
#    if str(value2)=="Regression":
#        if contents:
#            contents=contents[0]
#            filename=filename[0]
#            df=parse_contents(contents,filename)
#            if value1:
#                options=[{'label': name, 'value': name} for name in [str(lin(df,value1)[0])] ]
        #    value=[lin(df,value)]
#    return  options  
    




@app.callback(Output('acc', 'children'),
              [Input('cible', 'value')],[Input('predire', 'value')], [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output5(value1,variables,contents,value2,filename):
    children = html.Div()
    
    if "Regression" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if value1:
                if variables:
                    children=html.Div(["R square of Regression =",  str(lin(df,value1, variables)[0])])
     
    return children




# Decision tree classifier





@app.callback(Output('dtc_continue', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtc(value1,contents,value2,filename):
    
    children = html.Div()
    if "Decision tree Classifier" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            
            if value1:
                children=html.Div(["Accuracy_score of Decision Tree Classifier =",  str(dtc_continue(df, value1)[0])]) 
                               
     
    return children






# Decision tree regressor

@app.callback(Output('dtr_continue', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr(value1,variables,contents,value2,filename):
    
    children = html.Div()
    if "Decision tree Regressor" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            
            if value1:
                if variables:
                    children=html.Div(["R square of Decision Tree Regressor =",  str(dtr_continue(df, value1, variables)[0])]) 
                               
     
    return children







# Gradient Boosting


@app.callback(Output('neuron', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output8(value1,variables,contents,value2,filename):
    
    children = html.Div()
    if "SGB" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1:
                if variables:
                    children=html.Div(["R square of Gradient boosting =",  str(gradient(df, value1, variables)[0])]) 
                               
     
    return children


##############################################################################
#####################          REGRESSION LOGISTIQUE       ###################
##############################################################################


#Stocker le resultat de matric confusion dans un tableau 

def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+['Précision']+['Sensiblité']+['F1']+['Fréquence']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df


@app.callback(Output('reglog', 'children'),
              [Input('predire','value')],[Input('cible', 'value')] , [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_RL(variables,vcible,contents,value2,filename):
    children = html.Div()
    if "regression logistique" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if variables: 
                if vcible:
                    start=time()
                    y = y=df[str(vcible)]
                    X=df.loc[:,variables]
                    param_grid = [    
                                {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                                 'C' : np.logspace(-4, 4, 20),
                                 'max_iter' : [100, 1000,2500, 5000]
                                 }
                                ]
                    # split X and y into training and testing sets
                    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

                    # instantiate the model (using the default parameters)
                    logreg = LogisticRegression(multi_class="multinomial")
                    
                    #Hyperparametre
                    clf = GridSearchCV(logreg, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
                    
                    clf.fit(X_train,y_train)
                    acc=accuracy_score(y_test, clf.best_estimator_.predict(X_test))
                    y_pred=clf.best_estimator_.predict(X_test)
                    y_pred_proba = clf.best_estimator_.predict_proba(X_test)[::,1]
                    y_scores=clf.best_estimator_.predict_proba(X_test)
                    
                    #score=r2_score(y_test,clf.best_estimator_.predict(X_test))
                    
                    # import the metrics class
                    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
                    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                    
                    """
                    fig = px.area(
                    x=fpr, y=tpr,
                    title='ROC Curve (AUC={auc(fpr, tpr):.4f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        width=700, height=500
                        )
                    """
                    
                    # Evaluating model performance at various thresholds
                    
                   
                    
                    if y_scores.shape[1]>2:
                    
                    #fig=plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
                    #fig=sn.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
                    #fig=sn.heatmap(confusion_matrix, annot=True)
                        y_scores=clf.best_estimator_.predict_proba(X_test)
                        y_onehot = pd.get_dummies(y_test, columns=clf.best_estimator_.classes_)

                    # Create an empty figure, and iteratively add new lines
                    # every time we compute a new class
                        fig_ROC = go.Figure()
                        fig_ROC.add_shape(
                                    type='line', line=dict(dash='dash'),
                                    x0=0, x1=1, y0=0, y1=1
                                    )       
                    
                        for i in range(y_scores.shape[1]):
                            y_true = y_onehot.iloc[:, i]
                            y_score = y_scores[:, i]

                            fpr, tpr, _ = roc_curve(y_true, y_score)
                            auc_score = roc_auc_score(y_true, y_score)

                            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                            fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

                        fig_ROC.update_layout(
                                        xaxis_title='False Positive Rate',
                                        yaxis_title='True Positive Rate',
                                        yaxis=dict(scaleanchor="x", scaleratio=1),
                                        xaxis=dict(constrain='domain'),
                                        width=700, height=500
                                        )
                        fig_thresh=go.Figure()
                    else:
                         auc = metrics.roc_auc_score(y_test, y_pred_proba)
                         fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
                         df1 = pd.DataFrame({
                                        'False Positive Rate': fpr,
                                        'True Positive Rate': tpr
                                        }, index=thresholds)
                         df1.index.name = "Thresholds"
                         df1.columns.name = "Rate"
                         fig_ROC = px.area(
                                            x=fpr, y=tpr,
                                            title=f'Courbe de ROC (AUC={auc})',
                                            labels=dict(x='False Positive Rate', y='True Positive Rate'),
                                            width=500, height=500
                                            ) 
                         fig_ROC.add_shape(
                                type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1
                              )
                         fig_thresh = px.line(
                                    df1, title='TPR and FPR at every threshold',
                                    width=700, height=500
                                    )
                         fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
                         fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
                         
                          
                    indice=metrics.classification_report(y_test,y_pred)
                    indice=report_to_df(indice)
                    fig_hist = px.histogram(
                                            x=y_pred_proba, color=y_test, nbins=50,
                                            labels=dict(color='True Labels', x='Score')
                                            )
                    #Matrice de confusion
                    """
                    fig_matcon=plt.figure()
                    fig_matcon.add_subplot(111)
                    sn.heatmap(cnf_matrix,annot=True,square=True,cbar=False,fmt="d")
                    plt.xlabel("predicted")
                    plt.ylabel("true")
                    """
                    catego=clf.classes_
                    #fig_matcon=ff.create_annotated_heatmap(cnf_matrix)
                    fig_matcon=px.imshow(cnf_matrix,labels=dict(x="Prédiction", y="Observation", color="Nombre d'individus"),x=catego,y=catego,color_continuous_scale="Tealgrn",title="Analyse Discriminante Linéaire : Matrice de confusion")
                    end=time()
                    duration=(end-start)
                    return html.Div([
                                
                                  html.H2(
                                               children=f"Temps de calcul = {duration}",
                                               style={
                                                       'textAlign': 'center',
                                                       'color': colors['text']
                                                       }
                                                       ),
                                   html.Div(children=f"Taux d'erreur = {1-acc}", style={
                                                       'textAlign': 'center',
                                                       'color': colors['text']
                                                       }),
                                   html.Div([dash_table.DataTable(id='data-table',
                                                                   #title= f'Evaluation(Taux d''erreur={acc})',
                                                                   columns=[{"name": i, "id": i} for i in indice.columns],
                                                                   data=indice.to_dict('rows'),
                                                                   editable=True
                                                                   )]),
                                    html.Div([dcc.Graph(id='MaCo', figure=fig_matcon)]),
                                    html.Div([dcc.Graph(id='ROC', figure=fig_ROC)]),
                                    html.Div([dcc.Graph(id='Hist', figure=fig_hist)]),
                                    html.Div([dcc.Graph(id='Thresh', figure=fig_thresh)]) 
                                    ])


    """
html.Div(children=f"Score = {score}", style={
'textAlign': 'center',
'color': colors['text']
}),
"""

##############################################################################
###########################           ADL        #############################
##############################################################################

def calcul_adl(df,vcible,variables):
    y = df.loc[:,[str(vcible)]]
    X=df.loc[:,variables]
    pd.get_dummies(X)
    #découpage entrainement / test 
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, stratify=y)

    #instanciation
    lda = LinearDiscriminantAnalysis()
    
    start = time()
    
    #validation croisée 
    scores = cross_val_score(lda, X, y, cv=5)
    score_moyen = round(np.mean(scores),3)
    #apprentissage
    lda.fit(XTrain,yTrain)
    
    done = time()
    tps = round(done - start,3)
    
    #prediction 
    ypred = lda.predict(XTest)
    #matrice de confusion
    mc = metrics.confusion_matrix(yTest,ypred)
    
    #calcul des métriques par classe 
    met= metrics.classification_report(yTest,ypred,output_dict=True)
    #calcul du taux d'erreur 
    err = round(1-metrics.accuracy_score(yTest,ypred),3)
    
    #récupération des labels des classes 
    catego=lda.classes_
    
    return mc, err,catego,met,score_moyen,tps









# Mise à jour les données


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)

        table = html.Div([
            html.H5(filename)#,
         #   dash_table.DataTable(
         #       data=df.head(10).to_dict('rows'),
          #      columns=[{'name': i, 'id': i} for i in df.columns] #df.columns
          #  )#,
           # html.Hr(),
           # html.Div('Raw Content'),
           # html.Pre(contents[0:200] + '...', style={
            #   'whiteSpace': 'pre-wrap',
            #    'wordBreak': 'break-all'
            #})
        ])

    return table
              



# GRAPHE

#ADL


@app.callback(Output('graph_adl', 'children'),
              [Input('predire','value')],[Input('cible', 'value')] , [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_sortie_adl(variables,vcible,contents,value2,filename):
    figu=html.Div()
    if "Analyse Discriminante linéaire" in value2:
        if contents and variables:
            contents=contents[0]
            filename=filename[0]
    
            df=parse_contents(contents,filename)
            table,err,catego,met,score_moyen,tps =calcul_adl(df, vcible, variables)
            #test=""
            #for cat in catego:
             #   test=test+str(met[cat]["precision"])
             
            #récupération de la précision et du rappel par classe dans un vecteur
            met_classe=[]
            for cat in catego:
                met_classe.append(met[cat]["precision"])
                met_classe.append(met[cat]["recall"])
            
            met_classe=np.array(met_classe)
            #transformation du vecteur en matrice 
            met_classe=met_classe.reshape(len(catego),2)
            
            fig=px.imshow(table,labels=dict(x="Prédiction", y="Observation", color="Nombre d'individus"),x=catego,y=catego,color_continuous_scale="Tealgrn",title="Analyse Discriminante Linéaire : Matrice de confusion")
            fig2=px.imshow(met_classe,x=["précision","rappel"],y=catego,color_continuous_scale="Tealgrn",title="Analyse Discrimante Linéaire : Indicateurs par classe ")
            
           # figu=html.Div(children=["Analyse Discriminante Linéaire",dcc.Graph(id='figadl', figure=fig), "acc : "+str(acc)+"---test : "+str(met_classe)+"------ "+str(met),dcc.Graph(id='figadl2', figure=fig2)])
            figu=html.Div(children=[html.H6("Temps de calcul : "+str(tps)),html.H6("Score moyen : "+str(score_moyen)),dcc.Graph(id='figadl', figure=fig),"Taux d'erreur : ",str(err),dcc.Graph(id='figadl2', figure=fig2)])
         
                               
    return figu



# Arbre de decision : regressor





@app.callback(Output('graph', 'children'),
              [Input('cible', 'value')], [Input('predire','value')],[Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr_graph(value1,variables,contents,value2,filename):
    figu=html.Div()
    if "Decision tree Regressor" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    data_frame=dtr_continue(df, value1, variables)[1]
#                fig = px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                    figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Decision tree Regressor"))])
                               
    return figu




# Arbre de decision : classifier (matrice de confusion)





# Regression


@app.callback(Output('graph1', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output19(value1,variables,contents,value2,filename):
    figu=html.Div()
    if "Regression" in value2:   
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    data_frame=lin(df, value1, variables)[1]
                
  #              fig=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                    figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Regression"))])
         
                               
    return figu


# Gradient boosting


@app.callback(Output('graph2', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output29(value1,variables,contents,value2,filename):
    figu=html.Div()
    if str(value2)=="SGB":
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    data_frame=gradient(df, value1)[1]
#                fig = px.scatter(data_frame, x="valeur reel", y="valeur predict", title="SGB")
                    figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="SGB"))])
                               
    return figu





# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@app.callback(Output('graph_PCA', 'children'),
              [Input('cible', 'value')], [Input('pre_algo', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_output30(value1,value2,contents,filename):
    figu=html.Div()
    if contents:
        contents=contents[0]
        filename=filename[0]
        df=parse_contents(contents,filename) 
        df_bis=pd.get_dummies(df)
        if value1:
            if value2=="Quantitative":
                X=df_bis.drop(columns=[str(value1)])
                y=df_bis[str(value1)].to_numpy()
                sc = StandardScaler() 
                X_normalized = sc.fit_transform(X)  
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_normalized) 
                components=np.c_[components,y]
#            fig = px.scatter_3d(components, x=0, y=1, z=2, labels={'0':'PC1', '1':'PC2', '2':'valeur réel'})
                figu=html.Div(children=[dcc.Graph(id='fig1',  figure=px.scatter_3d(components, color=y, x=0, y=1, z=2, labels={'0':'PC1', '1':'PC2', '2':'valeur réel'}))])        
            if value2=="Qualitative":
                X=df.drop(columns=[str(value1)])
                y=df[str(value1)]
                X=pd.get_dummies(X)
                sc = StandardScaler() 
                X_normalized = sc.fit_transform(X)  
                pca = PCA(n_components=3)
                components = pca.fit_transform(X_normalized) 
#                components=np.c_[components,y]
#            fig = px.scatter_3d(components, x=0, y=1, z=2, labels={'0':'PC1', '1':'PC2', '2':'valeur réel'})
                figu=html.Div(children=[dcc.Graph(id='fig1',  figure=px.scatter_3d(components, color=df[str(value1)], x=0, y=1, z=2, labels={'0':'PC1', '1':'PC2', '2':'PC3'}))])
                
    return figu










### RESET DROPDOWN CIBLE



@app.callback(Output('cible', 'value'), [Input('cible', 'options')])
def callback11(value):
    return ""

@app.callback(Output('pre_algo', 'value'), [Input('pre_algo', 'options')])
def callback12(value):
    return ""


@app.callback(Output('algo', 'value'), [Input('algo', 'options')])
def callback13(value):
    return ""












if __name__ == '__main__':
    app.run_server(debug=True)
    