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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



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
        value=['data']
        ),
    
    html.Div(id='data'), 
    html.Div(id='acc'),
    html.Div(id='dt_continue'),
    html.Div(id='neuron'),
    html.Div(id='gradient_class'),
    html.Div(id='adl'),
    html.Div(id='ensemble'),
    
   
   
    ], style={'width': '30%', 'display': 'inline-block'}) ,
    
    
    
   html.Div([ 
    
     html.Div(id='graph3',),
    html.Div(id='graph',),
    html.Div(id='graph_c',),
    html.Div(id='graph1',),
    html.Div(id='graph2',),
    ],style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
    
  

],
   )





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
        ])

    return df
        


# Fonction qui détermine les variables qualitatives/quantitatives du CIBLE


def QT_function0(df,value):
    out=[]
    if str(df.dtypes[str(value)])=='object':
        out="Qualitative"
    else:
        out="Quantitative"
    
    return [str(out)]


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
        options=[{'label':name, 'value': name} for name in df.columns.tolist()]+[{'label':'data', 
                                                                                  'value':'data'}]
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
    





# REGRESSION LINEAIRE 

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold
import numpy as np


# recherche hyperparametre

def lin(df,value):
    c_space = np.logspace(-4, 0, 20)
    params = {'alpha': c_space}
    df_bis=pd.get_dummies(df)
    X=df_bis.drop(columns=[str(value)])
    y=df_bis[str(value)]
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
def gradient(df,value):
    params={'n_estimators':[16,32,64,100,200], 'learning_rate':[0.25,0.1,0.05,0.025],
            'max_depth':[1,2,4,8], 'subsample': [0.5,0.9,1], 'max_features':[0.5,0.75]}
    gb = GradientBoostingRegressor()
    df_bis=pd.get_dummies(df)
    X=df_bis.drop(columns=[str(value)])
    y=df_bis[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    gb_cv=GridSearchCV(gb, params, cv=KFold(5), n_jobs=-1)
    gb_cv.fit(X_train,y_train)
    y_pre=gb_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    score=r2_score(y_test, gb_cv.best_estimator_.predict(X_test))
    return [score, data_frame]
    
    
    
   











# DECISION TREE REGRESSION



# recherche hyperparametre

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
scoring = make_scorer(r2_score)

def dtr_continue(df,value):
    
    params = {"max_depth": [3,6,9,12, None],
              "min_samples_leaf": np.arange(1,9,1),
              "criterion": ["mse", "friedman_mse", "mae"]}
    df_bis=pd.get_dummies(df)
    X=df_bis.drop(columns=[str(value)])
    y=df_bis[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    dt = DecisionTreeRegressor()
    dt_cv=GridSearchCV(dt, params, cv=5, scoring=scoring, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=r2_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
    dict={'valeur réelle':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    return [acc, data_frame]
    


# DECISION TREE CLASSIFIER


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
scoring = make_scorer(r2_score)

def dtc_continue(df,value):
    
    params = {"max_depth": [3,6,9,12, None],
              "min_samples_leaf": np.arange(1,9,1),
              "criterion": ["mse", "friedman_mse", "mae"]}
    X=df.drop(columns=[str(value)])
    X=pd.get_dummies(X)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    dt = DecisionTreeClassifier()
    dt_cv=GridSearchCV(dt, params, cv=5, scoring=scoring, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=r2_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
    dict={'classe réelle':y, 'classe predicte': y_pre}
    data_frame=pd.DataFrame(dict)
    return [acc, data_frame]




    
    
    
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
              [Input('cible', 'value')], [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output5(value1,contents,value2,filename):
    children = html.Div()
    if "Regression" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if value1:
                children=html.Div(["R square of Regression =",  str(lin(df,value1)[0])])
     
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
                children=html.Div(["R square of Decision Tree Classifier =",  str(dtc_continue(df, value1)[0])]) 
                               
     
    return children





# Decision tree regressor

@app.callback(Output('dtr_continue', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr(value1,contents,value2,filename):
    
    children = html.Div()
    if "Decision tree Regressor" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            
            if value1:
                children=html.Div(["R square of Decision Tree Regressor =",  str(dtr_continue(df, value1)[0])]) 
                               
     
    return children







# Gradient Boosting


@app.callback(Output('neuron', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output8(value1,contents,value2,filename):
    
    children = html.Div()
    if "SGB" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1:
                children=html.Div(["R square of Gradient boosting =",  str(gradient(df, value1)[0])]) 
                               
     
    return children


#ADL

@app.callback(Output('adl', 'children'),
              [Input('predire','value')],[Input('cible', 'value')] , [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_ADL(variables,vcible,contents,value2,filename):
    children = html.Div()
    if "Analyse Discriminante linéaire" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if variables: 
                if vcible:
                    y = df.loc[:,[str(vcible)]]
                #X=df
                #del X[value]
                    X=df.loc[:,variables]
                    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, stratify=y)
                #XTrain = df.iloc[:,0:4]
                #yTrain = df.iloc[:,4]
                # X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, stratify=y)
                #instanciation
                    lda = LinearDiscriminantAnalysis()
                #apprentissage
                    lda.fit(XTrain,yTrain)
                #prediction
                    ypred = lda.predict(XTest)
                #matrice de confusion
                    mc = metrics.confusion_matrix(yTest,ypred)
                
                
                
                    children=html.Div(["var : "+str(variables)+" _____ "+str(lda.coef_)+" ---- matrice de confusion : "+str(mc)+" ---- Taux d'erreur : "+str(1.0-metrics.accuracy_score(yTest,ypred))+" ---- Sensibilité (rappel) et précision par classe : "+str(metrics.classification_report(yTest,ypred))])
     
    return children














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


# Arbre de decision : regressor





@app.callback(Output('graph', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr_graph(value1,contents,value2,filename):
    figu=html.Div()
    if str(value2)=="Decision tree regressor":
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                data_frame=dtr_continue(df, value1)[1]
                fig = px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                figu=html.Div(children=[dcc.Graph(figure=fig)])
                               
    return figu




# Arbre de decision : classifier





@app.callback(Output('graph_c', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtc_graph(value1,contents,value2,filename):
    figu=html.Div()
    if str(value2)=="Decision tree":
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                data_frame=dtc_continue(df, value1)[1]
                fig = px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                figu=html.Div(children=[dcc.Graph(figure=fig)])
                               
    return figu


# Regression


@app.callback(Output('graph1', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output19(value1,contents,value2,filename):
    figu=html.Div()
    if str(value2)=="Regression": 
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                data_frame=lin(df, value1)[1]
                
  #              fig=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique"))])
         
                               
    return figu


# Gradient boosting


@app.callback(Output('graph2', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output29(value1,contents,value2,filename):
    figu=html.Div()
    if str(value2)=="SGB":
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                data_frame=gradient(df, value1)[1]
                fig = px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Graphique")
                figu=html.Div(children=[dcc.Graph(figure=fig)])
                               
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
