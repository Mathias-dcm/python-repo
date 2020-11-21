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





#------------------------------------------------Dessiner layout-------------------------------------------------- 


app.layout = html.Div([
    
    html.H1("Interface d'analyse de données",style={ 'textAlign': 'center'}),
    #1ère colonne du layout 
    html.Div([ 
    
        #Glisser-Déposer 
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Glissez Déposez ou ',
                html.A('Chargez un fichier')
            ]),
            style={
                'width': '70%',
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
        
        #Affichage du nom du fichier chargé 
        html.Div(id='output-data-upload',
             style={ 'width': '50%'} 
        ),       
    
        dcc.Dropdown(
            id='cible',
            children=html.Div(['Choisir variable cible']),
            placeholder="Choix de la variable à prédire", 
            options=[{'label':'data', 'value': 'data'}],
            multi=False,
            style={'backgroundColor': '#84DCCF'},
            className='stockselector',
            #value=[]
        ),

        html.Div(
            id='pre_algo',
            children=["Type de la variable cible"],
            style={'backgroundColor': '#FFCB77'},
        ),

        dcc.Dropdown(
            id='algo',
            children=html.Div([ 'Choisir algo' ]),
            placeholder="Choix de l'algorithme",
            options=[ {'label':'data', 'value': 'data'}],
            multi=True,
            style={'backgroundColor': '#84DCCF'},
            className='stockselector',
            #value=[]
        ),

        dcc.Dropdown(
            id='predire',
            children=html.Div(['Choisir variables explicatives']),
            placeholder="Choix des variables prédictives", 
            style={'backgroundColor': '#FFCB77'},
            options=[ ],
            multi=True,
            className='stockselector',
            #        value=['data']
        ),
    
        html.Div(id='tabletype'),
        
        html.Div(id='data'),
  
        html.Div(id='graph_PCA',),
    
    #Fin de la première colonne
    ], style={'width': '30%', 'display': 'inline-block'}) ,
    
    #2è colonne 
    html.Div([ 

       #Création d'onglets pour afficher les résultats des différentes méthodes 
       html.Div(id='div_onglets', className='control-tabs', children=[
            dcc.Tabs(id='tabs_onglets', value='tabs', children=[
            
                #Onglet 1
                dcc.Tab(id='tab1', value='tab-1',children=[
                    html.Div(id='para', children=
                          [dcc.Dropdown(
                              id='parameter',
                              children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix des parametres pour la Régression (alpha)",
                              options=[{'label':'Le meilleur paramètre', 'value': 'Le meilleur paramètre'}]+[{'label':name, 'value': name} for name in list(np.logspace(-4,0,20))],
                              multi=False,
                              style={'backgroundColor': '#FFCAB1'},
                              className='stockselector',
                              #value=[]
                          ), ] , style= {'display': 'none'}
                    ),
                    
                    html.Div(id='paraRegLog_1', children=[
                        dcc.Dropdown(
                            id='paraRegLog1',children=
                            html.Div(['Choisir C']),
                            placeholder="Choix des parametres C",
                            options=[{'label':f'c={name}', 'value': name} for name in list(np.logspace(-4, 4, 20))],
                            multi=False,
                            style={'backgroundColor': '#FFCAB1'},
                            className='stockselector',
                        )], style= {'display': 'none'}),        
            
                     html.Div(id='paraRegLog_2', children=[
                         dcc.Dropdown(id='paraRegLog2',children=
                            html.Div(['Choisir pelnaty']),
                            placeholder="Choix de penalty",
                            options=[{'label':f'pelnaty={name}', 'value': name} for name in ['l2', 'none']],
                            multi=False,
                            style={'backgroundColor': '#FFCAB1'},
                            className='stockselector',
                         )], style= {'display': 'none'}),   
                     
                     html.Div(id='acc'),
                     html.Div(id='graph1',),
                     html.Div(id='reglog')
                ]),
                
            #Onglet 2  
            dcc.Tab(id="tab2", value='tab-2',children=[html.Div(id='param_dtr', children=
                        [dcc.Dropdown(id="radio_dtr",
                            options=[
                                {'label': name, 'value': name} for name in ['Paramètres optimaux', 'Paramètres manuels']],
                            value='Paramètres optimaux',
                            style={'width':'75%'}   
                        ),],style={'display': 'none'}),  
                                                       
                        html.Div(id='paradtr1',children=[
                            dcc.Dropdown(
                                id='depth_dtr',
                                #children=html.Div(['Choisir variable cible' ]),
                                placeholder="Choix du max_depth",
                                options=[{'label':name, 'value': name} for name in [3,6,9,12]],
                                multi=False,
                                style={'width':'75%'},
                                className='stockselector',
                            )
                        ],style= {'display': 'none'}),

                       html.Div(id='paradtr2',children=[
                           dcc.Dropdown(
                              id='sample_dtr',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du min_samples_leaf",
                              options=[{'label':name, 'value': name} for name in list(np.arange(1,9,1))],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                          )
                       ,],style={'display': 'none'}),
                        
                       html.Div(id='paradtr3',children=[
                           dcc.Dropdown(
                              id='criterion_dtr',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du criterion",
                              options=[{'label':name, 'value': name} for name in ["mse", "friedman_mse", "mae"]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                        ),], style={'display': 'none'}),
                    
                    
                    html.Div(id='param_dtc', children=
                        [dcc.Dropdown(id="radio_dtc",
                            options=[
                                {'label': name, 'value': name} for name in ['Paramètres optimaux', 'Paramètres manuels']],
                           
                            
                            value='Paramètres optimaux',
                            style={'width':'75%'}
                            
                        ),],style={'display': 'none'}),      
                    html.Div(id='paradtc1',children=[dcc.Dropdown(
                              id='depth_dtc',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du max_depth",
                              options=[{'label':name, 'value': name} for name in [3,6,9,12]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector',
                        ),],style= {'display': 'none'}),


                   html.Div(id='paradtc2',children=[dcc.Dropdown(
                              id='sample_dtc',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du min_samples_leaf",
                              options=[{'label':name, 'value': name} for name in list(np.arange(1,9,1))],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                        ),],style={'display': 'none'}),
                        
                    html.Div(id='paradtc3',children=[dcc.Dropdown(
                              id='criterion_dtc',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du criterion",
                              options=[{'label':name, 'value': name} for name in ["gini", "entropy"]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                        ),], style={'display': 'none'}),
                    
                 
                    
                    html.Div(id='dtr_continue'),
                    html.Div(id='dtc_continue'),
                    html.Div(id='graph_dtc'),
                    html.Div(id='graph',)
           
                ]),
            

            
            #Onglet 3
            dcc.Tab(id="tab3", value='tab-3',children=[
                
                #Contrôles pour les hyper paramètres de l'ADL 
                html.Div(id='param_adl', children=[
                        #Radio buttons ADL : optimal/manuel
                        dcc.RadioItems(id="radio_adl",
                            options=[
                                {'label': 'Paramètres optimaux', 'value': 'opti'},
                                {'label': 'Paramètres manuels', 'value': 'manu'}
                            ],
                            #par défaut on applique les paramètres optimaux 
                            #value='opti',
                            labelStyle={'display': 'inline-block'}
                        ),  
                        
                        #Dropdown choix du solver ADL 
                        dcc.Dropdown(
                              id='solver_adl',
                                  #children=html.Div(['Choisir variable cible' ]),
                                  placeholder="Choix du solver",
                             # options=[{'label':'svd', 'value': 'svd'}]+[{'label':'lsqr', 'value': 'lsqr'}]+[{'label':'eigen','value':'eigen'}],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector',
                        ),
                        
                        #Dropdown choix du shinkrage ADL
                        dcc.Dropdown(
                              id='shrinkage_adl',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du shrinkage",
                              #options=[{'label':'None', 'value': 'None'}]+[{'label':'auto', 'value': 'auto'}],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                        )
                        ] , style= {'display': 'none'}
                ),
                
                
                html.Div(id='param_sgb', children=
                        [dcc.Dropdown(id="radio_sgb",
                            options=[
                                {'label': name, 'value': name} for name in ['Paramètres optimaux', 'Paramètres manuels']],
                            value='Paramètres manuels',
                            style={'width':'75%'}   
                        ),],style={'display': 'none'}),  
                
                
                
                 html.Div(id='parasgb1',children=[
                            dcc.Dropdown(
                                id='n_estimators_sgb',
                                #children=html.Div(['Choisir variable cible' ]),
                                placeholder="Choix du n_estimators",
                                options=[{'label':name, 'value': name} for name in [16,32,64,100,200]],
                                multi=False,
                                style={'width':'75%'},
                                className='stockselector',
                            )
                        ],style= {'display': 'none'}),

                       html.Div(id='parasgb2',children=[
                           dcc.Dropdown(
                              id='learning_rate_sgb',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du learning_rate",
                              options=[{'label':name, 'value': name} for name in [0.25,0.1,0.05,0.025, 0.01]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                          )
                       ,],style={'display': 'none'}),
                        
                       html.Div(id='parasgb3',children=[
                           dcc.Dropdown(
                              id='max_depth_sgb',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du max_depth",
                              options=[{'label':name, 'value': name} for name in [1,2,4,8]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                           )
                        ,], style={'display': 'none'}),
                       
                       
                       html.Div(id='parasgb4',children=[
                           dcc.Dropdown(
                              id='sub_sample_sgb',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du sub_sample",
                              options=[{'label':name, 'value': name} for name in [0.5,0.9,1]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                           )
                        ,], style={'display': 'none'}),
                       
                       
                        html.Div(id='parasgb5',children=[
                           dcc.Dropdown(
                              id='max_features_sgb',
                              #children=html.Div(['Choisir variable cible' ]),
                              placeholder="Choix du max_features",
                              options=[{'label':name, 'value': name} for name in [0.5,0.75]],
                              multi=False,
                              style={'width':'75%'},
                              className='stockselector'
                           )
                        ,], style={'display': 'none'}),
                       
                
                
                
                html.Div(id='neuron'),
                html.Div(id='graph2',),
                html.Div(id='graph_adl')
            ]),
        ]),  
            #On inclut tous les éléments graphiques dont on a besoin 
            # html.Div(id='onglets_content',children=[
            #     html.Div(id='gradient_class'),
            #     html.Div(id='ensemble'),                  
            #     ])
        
    ]),
  ],style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
     
])    
#--------------------------------------Fin du layout------------------------------------------------- 



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
            type_var=QT_function0(df,cible)
            if "Qualitative" in type_var:
                return "Decision tree Classifier"
            else:
                return "Decision tree Regressor"
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



# Fonction qui détermine si la variable cible est qualitative ou quantitative 
def QT_function0(df,value):
    out=[]
    if str(df.dtypes[str(value)])=='object':
        out="Qualitative"
    else:
        if (len(np.unique(df[str(value)]))<6):
            out="Qualitative"
        else:
            out="Quantitative"
    return str(out)


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


#Affichage du type de la variable cible 
@app.callback(Output('pre_algo', 'children'),
              [Input('cible', 'value')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_output(value, contents,filename): 
    #child = []   

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if value:
            out=QT_function0(df, str(value))
            child=out
        else:child="Type de la variable cible"
    else: child="Type de la variable cible"
    return child


#Obtention liste choix des algo en fonction du type de la variable cible 
def QT_function(value):
    output=[]
#    if str(df.dtypes[str(value)])=='object':
#        out="Qualitative"
#    else:
#        out="Quantitative"
    if value=="Qualitative":
        output= ["Régression Logistique", "Decision tree Classifier","Analyse Discriminante Linéaire"]
    elif value=="Quantitative":
        output=["Regression", "SGB", "Decision tree Regressor"]
    else: 
        output=[]
    return output




# @app.callback(Output('algo', 'options'),
#               [Input('pre_algo', 'children')])

# def update_output00(value): 
#     options = []   

# #    if contents:
# #        contents = contents[0]
# #        filename = filename[0]
# #        df = parse_contents(contents, filename)
#     if value:
#         out=QT_function(str(value))
#         options=[{'label':name, 'value': name}  for name in out]
#     else:
#         options=[]
#     return options

#Mise à jour du dropdown avec le choix des algos
@app.callback(Output('algo', 'options'),
              [Input('cible', 'value')],[Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def update_algo(cible,contents,filename): 
    options = []   
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if cible:
            type_var=QT_function0(df,cible)
            out=QT_function(str(type_var))
            options=[{'label':name, 'value': name}  for name in out]
    return options



# Mise à jour dropdown du choix de la variable cible 
@app.callback(Output('cible', 'options'),
              [Input('upload-data', 'contents')], 
              [State('upload-data', 'filename')])

def update_variable_cible(contents, filename):
    options = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        options=[{'label':name, 'value': name} for name in df.columns.tolist()]#+[{'label':'data', 
                                                                                 # 'value':'data'}]
    return options
    
#Mise à jour dropdown du choix des variables explicatives 
@app.callback(Output('predire', 'options'),
              [Input('upload-data', 'contents'),
              Input('cible','value')], 
              [State('upload-data', 'filename')])

def update_variable_predire(contents, value,filename):
    options = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_contents(contents, filename)
        if value:
            del df[value]
        options=[{'label':name, 'value': name} for name in df.columns.tolist()]
    return options


# Mise à jour lAffichage du fichier chargé 
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
    else : 
        table = html.Div([
            html.H5("Aucune donnée")])

    return table
              




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
    


#-----------------------------------Gestion des calculs---------------------------------------------


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
    start=time()
    linear=Ridge(normalize=True)
    linear_cv=GridSearchCV(linear, params, cv=5)
    linear_cv.fit(X_train,y_train)
    end=time()
    y_pre=linear_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    score=r2_score(y_test, linear_cv.best_estimator_.predict(X_test))
    done=round(end-start,3)
    return [score, data_frame, done]
    
    

# hyperparametre fixé

def lin_bis(df,value,variables,para):
    c_space = [para]
    params = {'alpha': c_space}
#    df_bis=pd.get_dummies(df)
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    start=time()
    linear=Ridge(normalize=True)
    linear_cv=GridSearchCV(linear, params, cv=5)
    linear_cv.fit(X_train,y_train)
    end=time()
    y_pre=linear_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    score=r2_score(y_test, linear_cv.best_estimator_.predict(X_test))
    done=round(end-start,3)
    return [score, data_frame, done]





##############################################################################
###########################STOCHASTIC GRADIENT BOOSTING ######################
##############################################################################


from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
def gradient(df,value,variables):
    params={'n_estimators':[16,32,64,100,200], 'learning_rate':[0.25,0.1,0.05,0.025],
            'max_depth':[1,2,4,8], 'subsample': [0.5,0.9,1], 'max_features':[0.5,0.75]}
    gb = GradientBoostingRegressor()
 
    df_bis=df.loc[:,variables]   
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    start=time()
    gb_cv=GridSearchCV(gb, params, cv=KFold(5), n_jobs=-1)
    gb_cv.fit(X_train,y_train)
    end=time()
    y_pre=gb_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    done=round(end-start,3)
    score=r2_score(y_test, gb_cv.best_estimator_.predict(X_test))
    return [score, data_frame, done]
    
    
    
    
# hyperparametre fixe    
    
def gradient_bis(df,value,variables,para1,para2,para3,para4,para5):
    
    params = {'n_estimators': [para1],
              'learning_rate': [para2],
              'max_depth': [para3],
              'subsample': [para4],
              'max_features': [para5]}
    gb = GradientBoostingRegressor()
 
    df_bis=df.loc[:,variables]   
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    start=time()
    gb_cv=GridSearchCV(gb, params, cv=KFold(5), n_jobs=-1)
    gb_cv.fit(X_train,y_train)
    end=time()
    y_pre=gb_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    data_frame=pd.DataFrame(dict)
    done=round(end-start,3)
    score=r2_score(y_test, gb_cv.best_estimator_.predict(X_test))
    return [score, data_frame, done]
    
    
   
    
    
    
    
    


##############################################################################
########################### DECISION TREE REGRESSOR ##########################
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
    start=time()
    dt = DecisionTreeRegressor()
    dt_cv=GridSearchCV(dt, params, cv=5, scoring=scoring, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    end=time()
    acc=r2_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    done=round(end-start,3)
    data_frame=pd.DataFrame(dict)
    return [acc, data_frame, done]
    


def dtr_continue_params(df,value, variables,para1,para2,para3):
    
    params = {"max_depth": [para1],
              "min_samples_leaf": [para2],
              "criterion": [para3]}
   
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    start=time()
    dt = DecisionTreeRegressor()
    dt_cv=GridSearchCV(dt, params, cv=5, scoring=scoring, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    end=time()
    acc=r2_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X)
    dict={'valeur reel':y, 'valeur predict': y_pre}
    done=round(end-start,3)
    data_frame=pd.DataFrame(dict)
    return [acc, data_frame, done]


##############################################################################
########################### DECISION TREE CLASSIFIER ##########################
##############################################################################




from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
    start = time()
    dt_cv=GridSearchCV(dt, params, cv=5, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=accuracy_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X_test)
#    dict={'classe réelle':y, 'classe predicte': y_pre}
#    data_frame=pd.DataFrame(dict)

    done = time()
    tps = round(done - start,3)

#    matrice de confusion
    mc_dtc = metrics.confusion_matrix(y_test,y_pre)
    
    #calcul des métriques par classe 
    met_dtc = metrics.classification_report(y_test,y_pre,output_dict=True)
    
    #récupération des labels des classes 
    catego = dt_cv.classes_
    return [mc_dtc,catego,met_dtc,acc,tps]
   


def dtc_continue_params(df,value,variables,para1,para2,para3):
    
    params = {"max_depth": [para1],
              "min_samples_leaf": [para2],
              "criterion": [para3]}
    df_bis=df.loc[:,variables]
    X=pd.get_dummies(df_bis)
    y=df[str(value)]
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    dt = DecisionTreeClassifier()
    start = time()
    dt_cv=GridSearchCV(dt, params, cv=5, n_jobs=-1)
    dt_cv.fit(X_train,y_train)
    acc=accuracy_score(y_test, dt_cv.best_estimator_.predict(X_test))
    y_pre=dt_cv.best_estimator_.predict(X_test)
#    dict={'classe réelle':y, 'classe predicte': y_pre}
#    data_frame=pd.DataFrame(dict)

    done = time()
    tps = round(done - start,3)

#    matrice de confusion
    mc_dtc = metrics.confusion_matrix(y_test,y_pre)
    
    #calcul des métriques par classe 
    met_dtc = metrics.classification_report(y_test,y_pre,output_dict=True)
    
    #récupération des labels des classes 
    catego = dt_cv.classes_
    return [mc_dtc,catego,met_dtc,acc,tps]
     
    
# Regression    
    
    

    




@app.callback(Output('acc', 'children'),
              [Input('cible', 'value')], [Input('parameter','value')], [Input('predire', 'value')], [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_result_regression(value1,para,variables,contents,value2,filename):
    children = html.Div()
    
    if "Regression" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if value1:
                if variables:
                    if para:
                        if para=="Le meilleur paramètre":
                            children=html.Div([html.Div( ["Temps de calcul = ", str(lin(df,value1, variables)[2])]),html.Div(["R square of Regression =",  str(lin(df,value1, variables)[0])])])
                        else:
                            children=html.Div([html.Div( ["Temps de calcul = ", str(lin_bis(df,value1, variables,para)[2])]),html.Div(["R square of Regression =",  str(lin_bis(df,value1, variables,para)[0])])])
     
    return children 


# Decision tree classifier


@app.callback(Output('dtc_continue', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('radio_dtc','value')],[Input('depth_dtc','value')],[Input('sample_dtc','value')],[Input('criterion_dtc','value')],[Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])


def update_output_dtc(value,variables,params,para1,para2,para3,contents,value2,filename):
    
    children = html.Div()
    if "Decision tree Classifier" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            
            if value:
                if variables:
                    if params:
                        if params=="Paramètres optimaux": 
                            children=html.Div([html.Div(["Temps de calcul =", str(dtc_continue(df, value,variables)[4])]), html.Div([ "Accuracy of Tree Regressor =",  str(dtc_continue(df, value,variables)[3])])]) 
                        if params=="Paramètres manuels":
                            if para1:
                                if para2:
                                    if para3:
                                         children=html.Div([html.Div(["Temps de calcul =", str(dtc_continue_params(df, value,variables, para1,para2,para3)[4])]), html.Div([ "R square of Decision Tree Regressor =",  str(dtc_continue(df, value,variables)[3])])]) 
     
    return children





# Decision tree regressor

@app.callback(Output('dtr_continue', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('radio_dtr','value')],[Input('depth_dtr','value')],[Input('sample_dtr','value')],[Input('criterion_dtr','value')],[Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr(value1,variables,params, para1,para2,para3,contents,value2,filename):
    
    children = html.Div()
    if "Decision tree Regressor" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            
            if value1:
                if variables:
                    if params:
                        if params=="Paramètres optimaux": 
                            children=html.Div([html.Div(["Temps de calcul =", str(dtr_continue(df, value1, variables)[2])]), html.Div([ "R square of Decision Tree Regressor =",  str(dtr_continue(df, value1, variables)[0])])]) 
                        if params=="Paramètres manuels":
                            if para1:
                                if para2:
                                    if para3:
                                         children=html.Div([html.Div(["Temps de calcul =", str(dtr_continue_params(df, value1, variables, para1,para2,para3)[2])]), html.Div([ "R square of Decision Tree Regressor =",  str(dtr_continue_params(df, value1, variables,para1,para2,para3)[0])])]) 
                               
     
    return children




# Gradient Boosting


@app.callback(Output('neuron', 'children'),
              [Input('cible', 'value')], [Input('predire','value')],[Input('radio_sgb','value')],[Input('n_estimators_sgb','value')],[Input('learning_rate_sgb','value')],[Input('max_depth_sgb','value')],
              [Input('sub_sample_sgb','value')],  [Input('max_features_sgb','value')],
              [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_result_Boosting(value1,variables,params,para1,para2,para3,para4,para5,contents,value2,filename): 
    
    children = html.Div()
    if "SGB" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1:
                if variables:
                    if params:
                        if params=="Paramètres optimaux":
                            children=html.Div([html.Div(["Temps de calcul =", str(gradient(df, value1, variables)[2])]), html.Div(["R square of Gradient boosting =",  str(gradient(df, value1, variables)[0])])]) 
                        if params=="Paramètres manuels":
                            if para1:
                                if para2:
                                    if para3:
                                        if para4:
                                            if para5:
                                                children=html.Div([html.Div(["Temps de calcul =", str(gradient_bis(df, value1, variables,para1,para2,para3,para4,para5)[2])]), html.Div(["R square of Gradient boosting =",  str(gradient_bis(df, value1, variables,para1,para2,para3,para4,para5)[0])])]) 
                               
     
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
              [Input('predire','value')],
              [Input('paraRegLog1', 'value')],[Input('paraRegLog2', 'value')],
              [Input('cible', 'value')] , [Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_RL(variables,para1,para2,vcible,contents,value2,filename):
    if "Régression Logistique" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename)
            if variables: 
                if vcible:
                    start=time()
                    y = y=df[str(vcible)]
                    X=df.loc[:,variables]
                    if para1:
                        if para2:
                            param_grid = [    
                                {'penalty' : [para2],
                                 'C' : [para1],
                                 'solver' : ['lbfgs','newton-cg'],
                                 'max_iter' : [100, 1000,2500, 5000]
                                 }
                                ]
                        else:
                            param_grid = [    
                                {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                                 'C' : [para1],
                                 'solver' : ['lbfgs','newton-cg'],
                                 'max_iter' : [100, 1000,2500, 5000]
                                 }
                                ]
                            
                    else:
                        param_grid = [    
                            {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                             'solver' : ['lbfgs','newton-cg'],#,'newton-cg','sag','saga'],
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
                    score=clf.best_score_
                    
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
                                        xaxis_title='Taux de Faux Positif',
                                        yaxis_title='Taux de Vrai Positf',
                                        yaxis=dict(scaleanchor="x", scaleratio=1),
                                        xaxis=dict(constrain='domain'),
                                        width=700, height=500
                                        )
                        fig_thresh=go.Figure()
                    else:
                         auc = metrics.roc_auc_score(y_test, y_pred_proba)
                         fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
                         df1 = pd.DataFrame({
                                        'Taux de Faux Positif': fpr,
                                        'Taux de Vrai Positf': tpr
                                        }, index=thresholds)
                         df1.index.name = "Seuil"
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
                                    df1, title='Taux de TP et Taux de FP à chaque seuil',
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
                    fig_matcon=px.imshow(cnf_matrix,labels=dict(x="Prédiction", y="Observation", color="Nombre d'individus"),x=catego,y=catego,color_continuous_scale="Tealgrn",title="Matrice de confusion")
                    end=time()
                    duration=(end-start)
                    return html.Div([
                                
                                  html.H2(
                                               children=f"Temps de calcul = {duration}",
                                               style={
                                                       'textAlign': 'center',
                                                       'color': colors['text']
                                                       }),
                                  html.Div(children=f"Score = {score}", style={
                                                      'textAlign': 'center',
                                                      'color': colors['text']
                                                      }),
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
                                    html.Div([dcc.Graph(id='ROC', figure=fig_ROC),
                                              dcc.Graph(id='Thresh', figure=fig_thresh)
                                              ],style={'columnCount': 2}),
                                    html.Div([dcc.Graph(id='Hist', figure=fig_hist)]),
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

#Calcul de l'ADL avec les paramètres optimaux 
def calcul_adl(df,vcible,variables):
    
    y = df.loc[:,[str(vcible)]]
    X=df.loc[:,variables]
    pd.get_dummies(X)
    #découpage entrainement / test 
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, stratify=y)
    
    modele = LinearDiscriminantAnalysis() #Modèle
    params = {'solver':['svd', 'lsqr','eigen'], 'shrinkage':[None, 'auto']} #Paramètres à tester

    start = time()
    
    #instanciation - recherche des hyperparametres optimaux et validation croisee
    lda = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)
    
    #apprentissage
    lda.fit(XTrain,yTrain)
    
    #validation croisée 
    #scores = cross_val_score(lda, X, y, cv=5)
    #score_moyen = round(np.mean(scores),3)
   
    done = time()
    #Calcul du temps de calcul 
    tps = round(done - start,3)
    
    #Mean cross-validated score of the best_estimator
    score_moyen=round(lda.best_score_,3)
    
    #prediction 
    ypred = lda.predict(XTest)
    #matrice de confusion
    mc = metrics.confusion_matrix(yTest,ypred)
    
    #calcul des métriques par classe 
    met= metrics.classification_report(yTest,ypred,output_dict=True)
    met2= metrics.classification_report(yTest,ypred)
    #calcul du taux d'erreur 
    err = round(1-metrics.accuracy_score(yTest,ypred),3)
    
    #récupération des labels des classes 
    catego=lda.classes_
    
    return mc, err,catego,met,met2,score_moyen,tps


#Calcul de l'ADL avec les paramètres manuels 
def calcul_adl_manuel(df,vcible,variables,psolver,pshr):
    
    if pshr=="None":
        pshr=None
        
    y = df.loc[:,[str(vcible)]]
    X=df.loc[:,variables]
    pd.get_dummies(X)
    #découpage entrainement / test 
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, stratify=y)
    
    start = time()
    
    #instanciation 
    lda = LinearDiscriminantAnalysis(solver=psolver,shrinkage=pshr) 
    
    #apprentissage
    lda.fit(XTrain,yTrain)
    
    #validation croisée 
    scores = cross_val_score(lda, X, y, cv=5)
    score_moyen = round(np.mean(scores),3)
   
    done = time()
    #Calcul du temps de calcul 
    tps = round(done - start,3)
    
    #prediction 
    ypred = lda.predict(XTest)
    #matrice de confusion
    mc = metrics.confusion_matrix(yTest,ypred)
    
    #calcul des métriques par classe 
    met= metrics.classification_report(yTest,ypred,output_dict=True)
    met2= metrics.classification_report(yTest,ypred)
    #calcul du taux d'erreur 
    err = round(1-metrics.accuracy_score(yTest,ypred),3)
    
    #récupération des labels des classes 
    catego=lda.classes_
    
    return mc, err,catego,met,met2,score_moyen,tps


#------------------------------------------SORTIE DES ALGORITHMES--------------------------------------

#Sortie ADL
@app.callback(Output('graph_adl', 'children'),
              [Input('predire','value')],[Input('cible', 'value')] , [Input('solver_adl','value')],[Input('shrinkage_adl','value')],[Input('radio_adl','value')],[Input('upload-data', 'contents')],[Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_sortie_adl(variables,vcible,solver,shr,radio,contents,value2,filename):
    figu=html.Div()
    if "Analyse Discriminante Linéaire" in value2:
        if contents and variables:
                contents=contents[0]
                filename=filename[0]
    
                df=parse_contents(contents,filename)
            #if radio:
                #Choix manuel des paramètres 
                if radio == "manu": 
                    if solver=="svd":
                        table,err,catego,met,met2,score_moyen,tps =calcul_adl_manuel(df, vcible, variables,solver,None)
                    elif solver in ["lsqr","eigen"] and shr in ["None","auto"]:
                            table,err,catego,met,met2,score_moyen,tps =calcul_adl_manuel(df, vcible, variables,solver,shr)
                    else:
                        table,err,catego,met,met2,score_moyen,tps =calcul_adl(df, vcible, variables)
                    
                    #Choix automatique des paramètres 
                else:
                    table,err,catego,met,met2,score_moyen,tps =calcul_adl(df, vcible, variables)
             
                #récupération de la précision et du rappel par classe dans un vecteur : pour l'affichage graphique 
                met_classe=[]
                for cat in catego:
                    met_classe.append(met[cat]["precision"])
                    met_classe.append(met[cat]["recall"])
            
                met_classe=np.array(met_classe)
                #transformation du vecteur en matrice 
                met_classe=met_classe.reshape(len(catego),2)
            
                #récupération des métriques par classe pour affichage tableau 
                met2=report_to_df(met2)
            
                #Matrice de confusion graphique 
                fig=px.imshow(table,
                          labels=dict(x="Prédiction", y="Observation", color="Nombre d'individus"),
                          x=catego,y=catego,
                          color_continuous_scale="Tealgrn",
                          title="Matrice de confusion"
                )
            
                #Métriques par classe graphiques 
                fig2=px.imshow(met_classe,
                           labels=dict(color="Valeurs"),
                           x=["Précision","Rappel"],
                           y=catego,
                           color_continuous_scale="Tealgrn",
                           title="Indicateurs par classe "
                )
            
                #Sortie
                figu=html.Div(children=[
                    html.H5("Temps de calcul : "+str(tps)),
                
                    html.H6("Score moyen : "+str(score_moyen)),
                
                    "Taux d'erreur : ",str(err),
                
                    dash_table.DataTable(id='testmetadl',
                                     #title= f'Evaluation(Taux d''erreur={acc})',
                                     columns=[{"name": i, "id": i} for i in met2.columns],
                                     data=met2.to_dict('rows'),
                                     editable=True
                    ),
                
                    dcc.Graph(id='figadl', figure=fig),
                
                    dcc.Graph(id='figadl2', figure=fig2),
                
                    ],style={ 'textAlign': 'center'})
         
                               
    return figu





# Arbre de decision : regressor

@app.callback(Output('graph', 'children'),
              [Input('cible', 'value')], [Input('radio_dtr','value')],[Input('depth_dtr','value')],[Input('sample_dtr','value')],[Input('criterion_dtr','value')],
              [Input('predire','value')],[Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_output_dtr_graph(value1,params,para1,para2,para3,variables,contents,value2,filename):
    figu=html.Div()
    if "Decision tree Regressor" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    if params:
                        if params=='Paramètres optimaux':
                            data_frame=dtr_continue(df, value1, variables)[1]
                            figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Decision tree Regressor"))])
                        if params=='Paramètres manuels':
                            if para1:
                                if para2:
                                    if para3:
                                         data_frame=dtr_continue_params(df, value1, variables,para1,para2,para3)[1]
                                         figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Decision tree Regressor"))])
                            
    return figu




# Arbre de decision : classifier (matrice de confusion)





# Regression


@app.callback(Output('graph1', 'children'),
              [Input('cible', 'value')],[Input('parameter', 'value')], [Input('predire','value')], [Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_graph_Regression(value1,para,variables,contents,value2,filename):
    figu=html.Div()
    if "Regression" in value2:   
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    
                    if para:
                        if para=="Le meilleur paramètre":
                            data_frame=lin(df, value1, variables)[1]
                            figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Regression"))])
                        else:
                            data_frame=lin_bis(df, value1, variables,para)[1]
                            figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="Regression"))])
                               
    return figu


# Gradient boosting


@app.callback(Output('graph2', 'children'),
              [Input('cible', 'value')], [Input('predire','value')], [Input('radio_sgb','value')],[Input('n_estimators_sgb','value')],[Input('learning_rate_sgb','value')],[Input('max_depth_sgb','value')],
              [Input('sub_sample_sgb','value')],  [Input('max_features_sgb','value')],[Input('upload-data', 'contents')], [Input('algo', 'value')],
              [State('upload-data', 'filename')])

def update_graph_Boosting(value1,variables,params,para1,para2,para3,para4,para5,contents,value2,filename):
    figu=html.Div()
    if "SGB" in value2:
        if contents:
            contents=contents[0]
            filename=filename[0]
            df=parse_contents(contents,filename) 
            if value1: 
                if variables:
                    if params:
                        if params=='Paramètres optimaux':
                            data_frame=gradient(df, value1, variables)[1]
                            figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="SGB"))])
                        if params=='Paramètres manuels':
                            if para1:
                                if para2:
                                    if para3:
                                        if para4:
                                            if para5:
                                                data_frame=gradient_bis(df, value1, variables,para1,para2,para3,para4,para5)[1]
                                                figu=html.Div(children=[dcc.Graph(id='fig', figure=px.scatter(data_frame, x="valeur reel", y="valeur predict", title="SGB"))])
    return figu





#Représentation factorielle des données 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@app.callback(Output('graph_PCA', 'children'),
              [Input('predire','value')],[Input('cible', 'value')], [Input('pre_algo', 'children')], [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

<<<<<<< Updated upstream
def update_graph_ACP(variables,value1,value2,contents,filename):
=======
def update_graph_PCA(variables,value1,value2,contents,filename):
>>>>>>> Stashed changes
    figu=html.Div()
    if contents:
        contents=contents[0]
        filename=filename[0]
        df=parse_contents(contents,filename) 
        df_bis=pd.get_dummies(df)
        
        
        if value1:
            #Prise en compte des variables explicatives sélectionnées 
            if variables and len(variables)>1:
                X1=df.loc[:,variables]
            else:
                X1=df.drop(columns=[str(value1)])
            X=pd.get_dummies(X1)
                
            if value2=="Quantitative":
                #X=df_bis.drop(columns=[str(value1)])
                y=df_bis[str(value1)].to_numpy()
                sc = StandardScaler() 
                X_normalized = sc.fit_transform(X)  
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_normalized) 
                components=np.c_[components,y]
                fig = px.scatter_3d(components, x=0, y=1, z=2,color=2, labels={'0':'PC1', '1':'PC2', '2':'y observé'})
                fig2 = px.scatter(components, x=0, y=1, color=2,labels={'0':'PC1', '1':'PC2', '2':'y observé'})
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                features=X1.columns
                #Ajout de la sémantique des axes avec les variables d'origine 
                for i, feature in enumerate(features):
                    if str(X1.dtypes[str(feature)])!='object':
                        fig2.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=loadings[i, 0],
                        y1=loadings[i, 1]
                        )
                        fig2.add_annotation(
                        x=loadings[i, 0],
                        y=loadings[i, 1],
                        ax=0, ay=0,
                        xanchor="center",
                        yanchor="bottom",
                        text=feature,
                        )
               # figu=html.Div(children=[dcc.Graph(id='fig_acp2',  figure=fig2)])        
               # figu=html.Div(children=[dcc.Graph(id='fig_acp2',  figure=px.scatter(components, x=0, y=1, color=2,labels={'0':'PC1', '1':'PC2', '2':'y observé'})),dcc.Graph(id="testpca", figure=px.scatter_3d(components, x=0, y=1, z=2,color=2, labels={'0':'PC1', '1':'PC2', '2':'y observé'}))])        
                figu=html.Div(children=[dcc.Graph(id='fig_acp2', figure=fig2),dcc.Graph(id='fig_acp1',figure=fig)])
            if value2=="Qualitative":
                #X=df.drop(columns=[str(value1)])
                y=df[str(value1)]
                X=pd.get_dummies(X)
                sc = StandardScaler() 
                X_normalized = sc.fit_transform(X)  
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_normalized) 
#                components=np.c_[components,y]
#            fig = px.scatter_3d(components, x=0, y=1, z=2, labels={'0':'PC1', '1':'PC2', '2':'valeur réel'})

                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                features=X.columns
                fig=px.scatter(components, color=df[str(value1)], x=0, y=1, labels={'0':'PC1', '1':'PC2'})
                
                #Ajout de la sémantique des axes avec les variables d'origine 
                for i, feature in enumerate(features):
                    fig.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=loadings[i, 0],
                        y1=loadings[i, 1]
                        )
                    fig.add_annotation(
                        x=loadings[i, 0],
                        y=loadings[i, 1],
                        ax=0, ay=0,
                        xanchor="center",
                        yanchor="bottom",
                        text=feature,
                        )
                figu=html.Div(children=[dcc.Graph(id='fig1',  figure=fig)])
                
    return figu




######################## PARAMETRE ############################


@app.callback(Output('para', 'style'),
              [Input('algo', 'value')])

def update_output300(value):

    style={'display': 'none'}
    if "Regression" in value:
        style={'width': '33.5%','display': 'block'}

    return style




@app.callback(Output('param_dtr', 'style'),
              [Input('algo', 'value')])

def display_param_dtr(cible):

    style={'display': 'none'}
    if "Decision tree Regressor" in cible:
        style={'width': '37.5%','display': 'block'}

    return style


@app.callback(Output('paradtr1', 'style'),
              [Input('radio_dtr', 'value')], [Input('algo', 'value')])

def options_depth_dtr(value,algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Regressor" in algo:
            style={'width': '37.5%','display': 'block'}

    return style



@app.callback(Output('paradtr2', 'style'),
              [Input('radio_dtr', 'value')], [Input('algo', 'value')])

def options_sample_dtr(value, algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Regressor" in algo:
            style={'width': '37.5%','display': 'block'}
    return style




@app.callback(Output('paradtr3', 'style'),
              [Input('radio_dtr', 'value')], [Input('algo', 'value')])

def options_criterion_dtr(value,algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Regressor" in algo:
            style={'width': '37.5%','display': 'block'}
    return style




@app.callback(Output('param_dtc', 'style'),
              [Input('algo', 'value')])

def display_param_dtc(cible):

    style={'display': 'none'}
    if "Decision tree Classifier" in cible:
        style={'width': '37.5%','display': 'block'}

    return style

    style={'display': 'none'}
    if "SGB" in cible:
        style={'width': '37.5%','display': 'block'}

    return style

@app.callback(Output('paradtc1', 'style'),
              [Input('radio_dtc', 'value')], [Input('algo', 'value')])

def options_depth_dtc(value,algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Classifier" in algo:
            style={'width': '37.5%','display': 'block'}

    return style



@app.callback(Output('paradtc2', 'style'),
              [Input('radio_dtc', 'value')], [Input('algo', 'value')])

def options_sample_dtc(value, algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Classifier" in algo:
            style={'width': '37.5%','display': 'block'}
    return style




@app.callback(Output('paradtc3', 'style'),
              [Input('radio_dtc', 'value')], [Input('algo', 'value')])

def options_criterion_dtc(value,algo):

    style={'display': 'none'}
    if value=='Paramètres manuels':
        if "Decision tree Classifier" in algo:
            style={'width': '37.5%','display': 'block'}
    return style



@app.callback(Output('param_sgb', 'style'), 
              [Input('algo', 'value')]) 
 
def display_param_sgb(cible): 
 
    style={'display': 'none'} 
    if "SGB" in cible: 
        style={'width': '37.5%','display': 'block'} 
 
    return style 
 
 
 
@app.callback(Output('parasgb1', 'style'), 
              [Input('radio_sgb', 'value')], [Input('algo', 'value')]) 
 
def options_estimator_sgb(value,algo): 
 
    style={'display': 'none'} 
    if value=='Paramètres manuels': 
        if "SGB" in algo: 
            style={'width': '37.5%','display': 'block'} 
 
    return style 
 
 
 
 
@app.callback(Output('parasgb2', 'style'), 
              [Input('radio_sgb', 'value')], [Input('algo', 'value')]) 
 
def options_learningrate_sgb(value,algo): 
 
    style={'display': 'none'} 
    if value=='Paramètres manuels': 
        if "SGB" in algo: 
            style={'width': '37.5%','display': 'block'} 
    return style 
 
 
 
@app.callback(Output('parasgb3', 'style'), 
              [Input('radio_sgb', 'value')], [Input('algo', 'value')]) 
 
def options_depth_sgb(value,algo): 
 
    style={'display': 'none'} 
    if value=='Paramètres manuels': 
        if "SGB" in algo: 
            style={'width': '37.5%','display': 'block'} 
    return style 
 
 
 
@app.callback(Output('parasgb4', 'style'), 
              [Input('radio_sgb', 'value')], [Input('algo', 'value')]) 
 
def options_sample_sgb(value,algo): 
 
    style={'display': 'none'} 
    if value=='Paramètres manuels': 
        if "SGB" in algo: 
            style={'width': '37.5%','display': 'block'} 
    return style 
 
 
 
@app.callback(Output('parasgb5', 'style'), 
              [Input('radio_sgb', 'value')], [Input('algo', 'value')]) 
 
def options_features_sgb(value,algo): 
 
    style={'display': 'none'} 
    if value=='Paramètres manuels': 
        if "SGB" in algo: 
            style={'width': '37.5%','display': 'block'} 
    return style 




@app.callback(Output('param_adl', 'style'),
              [Input('algo', 'value')])

def display_param_adl(cible):

    style={'display': 'none'}
    if "Analyse Discriminante Linéaire" in cible:
        style=dict(display='flex',width='100%',columnCount= 2)#{'display': 'inline'}#'width': '70%',

    return style

@app.callback(Output('shrinkage_adl','options'),
             [Input('solver_adl','value')],[Input('radio_adl','value')])

def options_shrinkage_adl(solver,radio):
    if radio == "manu":
        if solver=="lsqr" or solver=="eigen":
            return [{'label':'None', 'value': 'None'}]+[{'label':'auto', 'value': 'auto'}]
    return []

@app.callback(Output('solver_adl','options'),
             [Input('radio_adl','value')])

def options_solver_adl(radio):
    if radio == "manu":
        return [{'label':'svd', 'value': 'svd'}]+[{'label':'lsqr', 'value': 'lsqr'}]+[{'label':'eigen','value':'eigen'}]
    else:
        return []


@app.callback(Output('paraRegLog_1', 'style'),
              [Input('algo', 'value')])


def update_outputParaRegLog1(value2):

    style={'display': 'none'}
    if ("Régression Logistique" in value2):
        style={'width': '33.5%','display': 'block'}

    return style

@app.callback(Output('paraRegLog_2', 'style'),
              [Input('algo', 'value')])

def update_outputParaRegLog(value2):

    style={'display': 'none'}
    if ("Régression Logistique" in value2):
        style={'width': '33.5%','display': 'block'}

    return style

### RESET DROPDOWN CIBLE



@app.callback(Output('cible', 'value'), [Input('cible', 'options')])
def callback11(value):
    return ""

# @app.callback(Output('pre_algo', 'children'), [Input('pre_algo', 'children')])
# def callback12(value):
#     return ""


@app.callback(Output('algo', 'value'), [Input('algo', 'options')])
def callback13(value):
    return ""



@app.callback(Output('parameter', 'value'), [Input('parameter', 'options')])
def callback14(value):
    return ""


@app.callback(Output('paraRegLog1', 'value'), [Input('paraRegLog1', 'options')])
def callback16(value):
    return ""

@app.callback(Output('paraRegLog2', 'value'), [Input('paraRegLog2', 'options')])
def callback17(value):
    return ""








if __name__ == '__main__':
    app.run_server(debug=True)
    