import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import data

# housing = pd.read_excel('/Users/smiths4/Documents/WhiteHat Training/Objectives work/London Rental Prices/ModelDF.xlsx')
housing = pd.read_excel('/ModelDF.xlsx')
housing = housing.drop(['Unnamed: 0'],axis=1)
housing





housing.describe()





# Create histogram bins of all variables to see spread of values

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=10, figsize=(20,15))
plt.show()

# Notice hows:
# 1) Values have different scales! May require adjusting





# Create a test set and set it aside
from sklearn.model_selection import train_test_split

# If i run the programme again, it will generate a new test set. Must set Random_state=0
# Keeps test set consistent across multiple runs

housing = housing.drop('SampleSize',axis=1)
housing_num = housing.set_index('Borough')
train_set, test_set = train_test_split(housing_num, test_size=0.2, random_state=0)




housing





df = train_set.copy()





corr_matrix = df.corr()
corr_matrix["MeanRent"].sort_values(ascending=False)




# Correlation dataframe
correlations = corr_matrix['MeanRent'].sort_values(ascending=False)
correlations = pd.DataFrame(data=correlations)
correlations.rename(columns = {'':"Factors",'MeanRent':'Correlation'}, inplace = True)
correlations['Characteristics'] = correlations.index

correlations = correlations.drop([
                                    'MeanRent'
#                                     ,'1YearGrowth'
#                                     ,'Pop'
                                    ], axis=0)
correlations['Characteristics'].replace(['HousePrice2017','MeanSalary','SocialHousing', 'GCSEscore','1YearGrowth','Pop','5YearGrowth']
                                   ,['House Prices','Mean Salary','% Social Housing', 'GCSE Performance','1 Yr Price Growth','Population','5 Yr Price Growth'],inplace=True)
correlations




# Graph 1 - Dash

fig_1 = go.Figure()

fig_1.add_trace(go.Bar(
    y=correlations['Correlation'],
    x=correlations['Characteristics'],
    text=correlations['Characteristics'],
    textposition='auto',
    marker = dict(color = '#8bcdcd'),
    hovertemplate =
    '<b>%{x}: </b><br>'+
    'Correlation with Mean Rent:<i> %{y} </i><br>'+
    '<extra></extra>',
    name='Correlation',
))

fig_1.update_layout(
                    title='LONDON BOROUGH CHARACTERISTICS<br>' + 'Correlation with Mean Rental Price'
                    ,xaxis=dict(
                               showgrid=False
                               ,showticklabels=False
                              ),
                    yaxis=dict(title='Correlation'),
                    width=950,
                    height=700,
                    legend= {'itemsizing': 'constant'},
                    font=dict(
                        family='Arial',
                        size=12,
                        color='#7f7f7f'))
fig_1.show()





# Graph 2 - Dash
fig_2 = go.Figure()

fig_2.add_trace(go.Scatter(
    x=housing['MeanRent'],
    y=housing['HousePrice2017'],
    text=housing['Borough'],
    mode='markers',
    marker = dict(color = '#FF6347',symbol='x'),
    hovertemplate =
    '<b>%{text}: </b><br>'+
    'Mean Rental Price:<i> %{x} </i><br>'+
    'Average House Price in 2017:<i> %{y} </i><br>'+
    '<extra></extra>',
    name='House Prices'))
    
    
fig_2.update_layout(
                    title='HOUSE PRICES v AVG RENT<br>'+ 'By London Borough',
                    xaxis=dict(title='Average Rent'
                               ,showgrid=False
                              ,tickprefix='£'),
                    yaxis=dict(title='Average House Price (2017)'
                                ,showgrid=False
                              ,tickprefix='£'),
                    width=500,
                    height=500,
                    legend= {'itemsizing': 'constant'},
#                     legend=dict(x=-.1),
                    font=dict(
                        family='Arial',
                        size=12,
                        color='#7f7f7f'))
fig_2.show()




# Graph 3 -Dash
fig_3 = go.Figure()

fig_3.add_trace(go.Scatter(
    x=housing['MeanRent'],
    y=housing['5YearGrowth']*100,
    text=housing['Borough'],
    mode='markers',
    marker = dict(color = '#FF6347',symbol='x'),
    hovertemplate =
    '<b>%{text}: </b><br>'+
    'Mean Rental Price:<i> %{x:£} </i><br>'+
    'Average House Price Growth:<i> %{y} </i><br>'+
    '<extra></extra>',
    name='House Prices'))
    
    
fig_3.update_layout(
                    title='HOUSE PRICE GROWTH v AVG RENT<br>'+ 'By London Borough',
                    xaxis=dict(title='Average Rental Price'
                               ,showgrid=False
                              ,tickprefix='£'),
                    yaxis=dict(title='Average 5 Year House Price Growth'
                                      ,showgrid=False
                                    ,ticksuffix='%'),
                    width=500,
                    height=500,
                    legend= {'itemsizing': 'constant'},
#                     legend=dict(x=-.1),
                    font=dict(
                        family='Arial',
                        size=12,
                        color='#7f7f7f'))
fig_3.show()





from pandas.plotting import scatter_matrix
attributes = ["MeanRent"
              , "MeanSalary"
              , "HousePrice2017"
              ,"Crimes/1000"]
# scatter_matrix(housing[attributes], figsize=(12, 8))





df = train_set.drop('MeanRent',axis=1)
df_labels = train_set['MeanRent'].copy()





# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_prepared = sc.fit_transform(df)





# LINEAR REGRESSION

# Train the model and predict on train set

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(df_prepared, df_labels)
some_data = df_prepared[:5]
some_labels = df_labels[:5]
print("Predictions:", lin_reg.predict(some_data))
print("Labels:", list(some_labels))





lin_reg.coef_
lin_reg.intercept_





# Linear Regression equation

coeffs = (list(zip(lin_reg.coef_.round(2), df.columns)))
interc = ((lin_reg.intercept_.round(2), 'Intercept'))
coeffs = pd.DataFrame(coeffs)
coeffs.rename(columns = {0: 'Coefficient'
                        ,1: 'Variable'},inplace=True)

interc = pd.DataFrame([interc]
                      , columns=(['Coefficient','Variable'])
                     )
coeffs = coeffs.append(interc,ignore_index=True)
coeffs





# Assess success of linear regression on train set
from sklearn.metrics import mean_squared_error
df_predictions = lin_reg.predict(df_prepared)
lin_mse = mean_squared_error(df_labels, df_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

average = np.mean(df_labels)
pc = (lin_rmse / average)*100
# print(lin_rmse)
# print(average)
# print(pc)





# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(df_prepared, df_labels)

df_predictions = tree_reg.predict(df_prepared)
tree_mse = mean_squared_error(df_labels, df_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse




# Cross Validation. 
# Comparison of decision tree model scores when running the model 10 times. 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, df_prepared, df_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
tree_mean = scores.mean()





# Comparison of linear reg model scores when running the model 10 times. 

lin_scores = cross_val_score(lin_reg, df_prepared, df_labels,
                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)




# Comparison of Random Forest scores when running the model 10 times. 

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(df_prepared, df_labels)

forest_scores = cross_val_score(forest_reg, df_prepared, df_labels,
                             scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)





# Table of model results for Dash


linear = str(lin_rmse_scores.round(2)).strip('[]')
linear = linear.replace('   ',' ')
linear = linear.replace('  ',' ')
linear = linear.replace(' ',', ')

tree = str(tree_rmse_scores.round(2)).strip('[]')
tree = tree.replace('   ',' ')
tree = tree.replace('  ',' ')
tree = tree.replace(' ',', ')

forest = str(forest_rmse_scores.round(2)).strip('[]')
forest = forest.replace('   ',' ')
forest = forest.replace('  ',' ')
forest = forest.replace(' ',', ')


train_res = {'':['Scores','MSE','RMSE'],
        '':['Scores','MSE','RMSE'],
        'Linear Regression':  [linear,lin_rmse_scores.mean().round(2),lin_rmse_scores.std().round(2)],
        'Decision Tree': [tree,tree_rmse_scores.mean().round(2),tree_rmse_scores.std().round(2)],
        'Random Forest': [forest,forest_rmse_scores.mean().round(2),forest_rmse_scores.std().round(2)],
        }

scores_table = pd.DataFrame (train_res
                        , columns = ['Measure','Linear Regression','Decision Tree','Random Forest']
                       )
Measure = ['Scores','MSE','RMSE']
scores_table['Measure'] = Measure
# scores_table


# GRID SEARCH - LinReg

from sklearn.model_selection import GridSearchCV

parameters = {'fit_intercept':[True,False], 'copy_X':[True,False], 'normalize':[True,False]}

lm = LinearRegression()
grid_search = GridSearchCV(lm, parameters, cv=3)
grid_search.fit(df_prepared, df_labels)

print(grid_search.best_estimator_)
print ("r2 / variance : ", grid_search.best_score_)


# Test Set predictions from LinReg
final_model = grid_search.best_estimator_

X_test = test_set.drop('MeanRent', axis=1)
y_test = test_set['MeanRent'].copy()
X_test_prepared = sc.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# print(final_rmse)
# print(final_predictions)
# print(y_test)

results = pd.DataFrame(y_test)
results['Predictions'] = final_predictions
results.reset_index(inplace=True)
results = results.sort_values('MeanRent',ascending=True)
results



# Performance of Linear Regression

final_rmse = round(final_rmse)
final_rmse
RMSE = ('The Root Mean Squared Error in predicting avg rent is £', final_rmse)

average = np.mean(results['MeanRent'])
pc = (final_rmse / average) * 100
pc = '{:,.2f}%'.format(pc)
pc

PCDiff = ('with the percentage difference between average rent & RMSE being ', pc)




results = pd.DataFrame(y_test)
results['Predictions'] = final_predictions
results.reset_index(inplace=True)
results = results.sort_values('MeanRent',ascending=True)
results





# VISUALISING Linear Regression results

fig_LR = go.Figure()

trace1 = fig_LR.add_trace(go.Bar(
    y=results['MeanRent'],
    x=results['Borough'],
    marker = dict(color = '#8bcdcd'),
    hovertemplate =
    'Actual Rent:<i> %{y} </i>',
    name='Actual',
))

trace2 = fig_LR.add_trace(go.Bar(
    y=results['Predictions'],
    x=results['Borough'],
    marker = dict(color = '#f9d56e'),
    hovertemplate =
    'Model Predicted Rent:<i> %{y} </i>',
    name='Predicted',
))

fig_LR.update_layout(
                    title='LINEAR REGRESSION RESULTS<br>' + 'Predicting Average Rent by London Borough'
                    ,xaxis=dict(title='TEST SET'
                               ,showgrid=False
#                                ,showticklabels=False
                              ),
                    yaxis=dict(title='Average Rental Price'
                              ,showgrid=False),
                    width=1000,
                    height=500,
                    legend= {'itemsizing': 'constant'},
                    font=dict(
                        family='Arial',
                        size=12,
                        color='#7f7f7f'))
fig_LR.show()





# # Grid Search - Random Forest


# param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#               {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

# forest_reg = RandomForestRegressor()

# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error')

# grid_search.fit(df_prepared, df_labels)

# grid_search.best_params_

# grid_search.best_estimator_

# RF_final_model = grid_search.best_estimator_
# RF_X_test = test_set.drop('MeanRent', axis=1)
# RF_y_test = test_set['MeanRent'].copy()
# RF_X_test_prepared = sc.transform(RF_X_test)
# RF_final_predictions = RF_final_model.predict(RF_X_test_prepared)
# RF_final_mse = mean_squared_error(RF_y_test, RF_final_predictions)
# RF_final_rmse = np.sqrt(RF_final_mse)
# print(RF_final_predictions)
# RF_y_test

# RF_results = pd.DataFrame(RF_y_test)
# RF_results['Predictions'] = RF_final_predictions
# RF_results.reset_index(inplace=True)
# RF_results = RF_results.sort_values('MeanRent',ascending=True)
# RF_results





average = np.mean(results['MeanRent'])
pc = (final_rmse / average) * 100
pc = '{:,.2f}%'.format(pc)
pc

PCDiff = ('with the percentage difference between average rent & RMSE being ', pc)




import dash
import dash_table

# App style
import dash_bootstrap_components as dbc

# App layout & interacitivity
import dash_html_components as html
import dash_core_components as dcc

# external stylesheets??
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://www.google-analytics.com/analytics.js']
external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#FFFFFF',
    'text': '#114B5F'
}

app.layout = html.Div(style={'backgroundColor': colors['background']},
                      children=[
                        html.H1(children='London Rental Price Model'
                                ,style={'textAlign':'center'
                                      ,'color':colors['text']
                                      ,'padding-top':'3%'}
                               ),
                        html.Div(children='''Predicting the average Borough rental price using Borough demographics & characteristics'''
                                ,style={'textAlign':'center'
                                       ,'color':colors['text']
#                                        ,'padding-top':'1%'
                                       }
                                ),
                        html.H4(children='''VARIABLES'''
                                ,style={'textAlign':'center'
                                       ,'color':colors['text']
                                       ,'padding-top':'1%'}
                                ),
                        dcc.Graph(
                            id='corr_values',
                            figure=fig_1
                                ,style={'width': '80%'
                                        ,'padding-left':'10%'
                                        , 'padding-right':'10%'}
                                ),
                          
                        html.Div(children='There is strong linear correlation between house prices and rent prices.'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']}
                                ),
                          
                        html.Div(children='Linear correlation with house price growth over 5 years is considerably weaker:'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-bottom':'2%'}
                                ),
                #   Two graphs, side by side          
                        html.Div([
                            html.Div([
                                html.H3(''),
                                dcc.Graph(id='fig_2'
                                         ,figure=fig_2
                                         ,style={'width': '40%','padding-left':'30%', 'padding-right':'5%'})
                            ], className="correlation"),

                            html.Div([
                                html.H3(''),
                                dcc.Graph(id='fig_3'
                                         ,figure=fig_3
                                         ,style={'width': '40%','padding-left':'35%', 'padding-right':'0%'})
                            ], className="correlation"),
                        ], className="row"
                        
                                ),
                        html.H4(children='MODEL EVALUATION'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                        ,'padding-bottom':'2%'}
                                ),                  
                          
                        html.Div(children='Testing 3 different models using these explanatory values, we compare their accuracy in predicting'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                       }
                                ),
                          
                        html.Div(children='the training data values. We find that Linear Regression is the most successful:'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                          
                        dash_table.DataTable(
                                        id='scores_table',
                                        columns=[{"name": i, "id": i} for i in scores_table.columns],
                                        data=scores_table.to_dict('rows'),
                                        style_header={'backgroundColor': '#8bcdcd',
                                                    'fontWeight': 'bold'
                                                     ,'textAlign':'center'},
                                        style_table={'maxWidth': '70%'
                                                    ,'padding-left':'22%'},
                                        style_data={'whiteSpace': 'normal',
                                                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                                    'height': 'auto',
                                                    'textAlign':'center'}
                                            ),
                        html.Div(children='To see how to the linear regression estimates MeanRent, we observe the intercept and variable coefficients:'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                          
                        dash_table.DataTable(
                                        id='coeff_table',
                                        columns=[{"name": i, "id": i} for i in coeffs.columns],
                                        data=coeffs.to_dict('rows'),
                                        style_header={'backgroundColor': '#8bcdcd',
                                                    'fontWeight': 'bold'
                                                     ,'textAlign':'center'},
                                        style_table={'maxWidth': '50%'
                                                    ,'padding-left':'36%'},
                                        style_data={'whiteSpace': 'normal',
                                                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                                    'height': 'auto',
                                                    'textAlign':'center'}
                                            ),
                          
                        html.Div(children='The intercept is the average MeanRent from the training set (1628). We interpret the remaining coefficients as:'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                       }
                                ),
                        html.Div(children='For a unit increase in MeanSalary, the MeanRent increases by 199.'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                       }
                                ),
                          
                        html.Div(children='Employing GridSearch to find the best combination of parameters, we find that this is the same as the default setting'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                       }
                                ),
                        html.Div(children='for Linear Regression. Comparing the predicted and actual values when running  the optimal model on the test set:'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                        html.H4(children='Linear Regression'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                        dcc.Graph(
                            id='fig_LR',
                            figure=fig_LR
                                ,style={'width': '80%'
                                        ,'padding-left':'10%'
                                        , 'padding-right':'10%'
                                       }
                                ),                  
                          html.Div(children=RMSE
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                       ,'padding-top':'1%'
                                       }
                                ), 
                        html.Div(children=PCDiff
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                        html.H4(children='Evaluation'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-top':'2%'
                                        ,'padding-bottom':'2%'
                                       }
                                ),
                        html.Div(children='This is an okay prediction! But there is some margin of error and there may be several reasons for this, including the (very)'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                       }
                                ),
                        html.Div(children='small dataset not providing enough to learn from. It is also possible that another regressor would have performed better'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                       }
                                ),
                         html.Div(children='on the test set.  Nonetheless, this was a good example to understanding the basics of machine learning with sci-kit learn.'
                                ,style={'textAlign':'center'
                                        ,'color':colors['text']
                                        ,'padding-bottom':'4%'
                                       }
                                ),
                          
                         dcc.Link('Github Code', href='https://github.com/susiesmith15/LondonRentalMarket'
                                 ,style={'textAlign':'center'}),
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.run_server(debug=False)

