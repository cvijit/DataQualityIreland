import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import json
from deepchecks.tabular.suites import full_suite
import plotly.figure_factory as ff
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
import sympy as smp
import seaborn as sns
def mento(data,f,i):
    y,x,s,m = smp.symbols("x s m y")
    fs = smp.integrate(f,(x,0,y)).doit()
    Fn = smp.lambdify((y,s,m),fs)
    fn = smp.lambdify((x,s,m),f)
    s=data[i].std()
    m=(data[i]).mean()
    x = np.linspace(min(data[i]),max(data[i]),len(data[i]))
    f = fn(x,s,m)
    F = Fn(x,s,m)
    us = np.random.rand(len(data))
    F_inv = x[np.searchsorted(F[:-1],us)]
    return F_inv
def pandas_profiling_report(df):
    df_report = ProfileReport(df,explorative=True)
    return df_report
def read_csv(source_data):
    df = pd.read_csv(source_data)
    return df 
def read_excel(source_data):
    df = pd.read_excel(source_data)
    return df
def OLS(df,S1):
    train=df.drop([S1],axis=1)
    test = df[S1]
    constant = sm.add_constant(train)
    model = sm.OLS(list(test),constant)
    result = model.fit()
    #new_constant=sm.add_constant(train)
    pred = result.predict()
    return pred
def main():
    df = None
    with st.sidebar.header("Source Data Selection"):
        selection = ["csv",'excel']
        selected_data = st.sidebar.selectbox("Please select your dataset fromat:",selection)
        if selected_data is not None:
            if selected_data == "csv":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.csv) data", type = ["csv"])
                if source_data is not None: 
                    df = read_csv(source_data)
            elif selected_data == "excel":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.xlsx) data", type = ["xlsx"])
                if source_data is not None:
                    df = read_excel(source_data)
        
       
    
    #if source_data is not None:
        #df = read_csv(source_data)
        st.header("Dataset")
        
    if df is not None:
        user_choices = ['Dataset Sample',"Data Quality",'Data Prediction']
        selected_choices = st.sidebar.selectbox("Please select your choice:",user_choices)
        
        if selected_choices is not None:
            if selected_choices == "Dataset Sample":
                st.info("Select dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns.")
                st.write(df)  
            elif selected_choices == "Data Prediction":
                choices = ['Ordinary Least Squares','Monte Carlo Simulation','interpolation']
                old_val = st.sidebar.selectbox(" ",choices,key=f"MyKey{1}")
                if old_val == "Ordinary Least Squares": 
                    st.markdown("Ordinary Least Squares")
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{2}")
                    a = []
                    for i in select:
                        if df[i].dtypes != object:
                            a.append(1)
                        else:
                            a.append(0)
                    if selection is not None:
                        for i in select:
                            if selection == i and all(a)==1:
                                data = OLS(df,i)
                                select_data1 = {i:data,"index":np.arange(len(data)),"color":"OLS"}
                                select_data1 = pd.DataFrame(select_data1)
                                select_data2 = {i:df[i],"index":np.arange(len(data)),"color":"Real"}
                                select_data2 = pd.DataFrame(select_data2)
                                select_data = {"OLS":data,"real":df[i]}
                                select_data = pd.DataFrame(select_data)
                                data1 = [select_data2,select_data1]
                                result = pd.concat(data1)
                                base = alt.Chart(result).mark_rule().encode( 
                                    x=alt.X('index', axis=alt.Axis( )),
                                    y=alt.Y(i,axis=alt.Axis( )),
                                    color = "color").properties(
                                    width=500,
                                    height=400,   
                                    ).interactive()
                                fig = px.scatter(result,x="index",y=i,color="color")
                                #figs = px.bar(result,x="index",y=i,color="color")
                                score = pearsonr(select_data["OLS"],select_data["real"])
                                #ax.legend("a","b")
                                tab1,tab2 = st.tabs(["Scatter plot theme", "Histogram theme"])
                                with tab1:
                                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                with tab2:
                                    #st.plotly_chart(figs, theme=None, use_container_width=True)
                                    st.altair_chart(base, use_container_width=True)
                                
                                with st.expander("See the OLS and Real data"):
                                    st.write(select_data)
                                st.write("The correlation coefficient between Real and OLS is",score[0])
                                st.write("The corresponding p-value is",score[1])
                            elif selection == i:
                                st.error("OLS only work with dataset which only include int and float data")
                                
                elif old_val == "interpolation":
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{3}")
                    for i in select:
                        if selection==i and df[i].dtypes!=object:
                            df_new = df[1:]
                            df_new = df_new[:-1]
                            train,test=train_test_split(df_new,test_size=0.25,train_size=0.75)
                            bb = df.loc[df.index==0]
                            train = pd.concat([bb,train])
                            train.loc[max(df.index),:]=df.iloc[max(df.index)]
                            x = np.array(train.index)
                            y = np.array(train[i])
                            f = interp1d(x,y,kind="cubic")
                            new_value = []
                            for iii in test.index:
                                new_value.append(f(iii))
                            select_data1 = {i:new_value,"index":np.arange(len(test.index)),"color":"interp1d"}
                            select_data1 = pd.DataFrame(select_data1)
                            select_data2 = {i:list(test[i]),"index":np.arange(len(test.index)),"color":"Real"}
                            select_data2 = pd.DataFrame(select_data2)
                            select_data = {"interp1d":list(map(float, new_value)),"real":list(map(float, test[i]))}
                            select_data = pd.DataFrame(select_data)
                            data1 = [select_data2,select_data1]
                            result = pd.concat(data1)
                            score = pearsonr(select_data["interp1d"],select_data["real"])
                            fig = px.scatter(result,x="index",y=i,color="color")
                            tab1,tab2 = st.tabs(["Scatter plot theme", "Line chart theme"])
                            with tab1:
                                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                            with tab2:
                                st.line_chart(select_data)
                                #st.plotly_chart(fig, use_container_width=True)
                                #st.altair_chart(base, use_container_width=True)
                                
                            with st.expander("See the interpolation and Real data"):
                                st.write(select_data)
                            st.write("The correlation coefficient between Real and interpolation is",score[0])
                            st.write("The corresponding p-value is",score[1])
                            
                elif old_val == "Monte Carlo Simulation":
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{3}")
                    for i in select:
                        if selection==i and df[i].dtypes!=object:
                            y,x,s,m = smp.symbols("x s m y")
                            f = 1/(s*(np.pi*2)**(1/2))*smp.exp(-(x-m)**2/(2*s**2))
                            data = mento(df,f,i)
                            select_data1 = {i:data,"index":np.arange(len(data)),"color":"Monte Carlo"}
                            select_data1 = pd.DataFrame(select_data1)
                            select_data2 = {i:df[i],"index":np.arange(len(data)),"color":"Real"}
                            select_data2 = pd.DataFrame(select_data2)
                            select_data = {"Monte Carlo":data,"real":df[i]}
                            select_data = pd.DataFrame(select_data)
                            data1 = [select_data2,select_data1]
                            result = pd.concat(data1)
                            hist_data = [select_data["Monte Carlo"], select_data["real"]]
                            group_labels = ['Monte Carlo', 'Real']
                            figs = ff.create_distplot(
                            hist_data, group_labels, bin_size=[0.2, .25, .5])
                            
                            
                            
                            base = alt.Chart(result).mark_rule().encode( 
                            x=alt.X('index', axis=alt.Axis( )),
                            y=alt.Y(i,axis=alt.Axis( )),
                            color = "color").properties(
                                    width=500,
                                    height=400,   
                                    ).interactive()
                            fig = px.scatter(result,x="index",y=i,color="color")
                                #figs = px.bar(result,x="index",y=i,color="color")
                            score = pearsonr(select_data["Monte Carlo"],select_data["real"])
                                #ax.legend("a","b")
                            tab1,tab2 = st.tabs(["Scatter plot theme", "Distrubtion theme"])
                            with tab1:
                                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                            with tab2:
                                st.plotly_chart(figs, use_container_width=True)
                                #st.altair_chart(base, use_container_width=True)
                                
                            with st.expander("See the Monte Carlo and Real data"):
                                st.write(select_data)
                            st.write("The correlation coefficient between Real and Monte Carlo is",score[0])
                            st.write("The corresponding p-value is",score[1])
                            
                
             
            elif  selected_choices == "Data Quality":
                box = ["Overview","Score","Data types","Descriptive statistics","Missing values","Duplicate records",
                     "Correlation", "Outliers","Data distribution"]
                selection = st.selectbox("Data Quality Selection",box,key=f"MyKey{4}") 
                if selection is not None:
                    if selection == "Overview":
                        df_report = pandas_profiling_report(df)
                        st.write("Profiling")
                        st_profile_report(df_report)
                    elif selection == "Data types":
                        types = pd.DataFrame(df.dtypes)
                        
                        a = types.astype(str)
                        st.dataframe(a)
                    elif selection == "Descriptive statistics":
                        types = pd.DataFrame(df.describe()).T
                        
                        a = types.astype(str)
                        st.table(a)
                    elif selection == "Missing values":
                        df.replace(0, np.nan, inplace=True)
                        types = pd.DataFrame(df.isnull().sum())
                        
                        a = types.astype(str)
                        st.write(a)
                        box = df.keys()
                        se = st.selectbox("Show missing values",box,key=f"MyKey{5}")
                        for i in box:
                            if se == i:
                                st.write(df[pd.isnull(df[i])])
                        
                    elif selection == "Duplicate records":
                        types = df[df.duplicated()]
                        
                        a = types.astype(str)
                        st.write("The number of duplicated rows is ",len(types))
                        st.write(a)
                        
                    elif selection == "Outliers":
                        fig = plt.figure(figsize=(4,3))
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check",box,key=f"MyKey{5}")
                        for i in box:
                            if se == i and df[i].dtypes !=object:
                                
                                sns.boxplot(df[i])
                                st.pyplot(fig)
                    elif selection == "Data distribution":
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check",box,key=f"MyKey{6}")
                        for i in box:
                            if se == i and df[i].dtypes !=object:
                                fig = plt.figure(figsize=(4,3))
                                sns.histplot(data = df,x=i,binwidth=3)
                                st.pyplot(fig)
                    elif selection == "Correlation":
                        fig,ax = plt.subplots()
                        sns.heatmap(df.corr(),annot = True,ax=ax)
                        st.pyplot(fig)
                    elif selection == "Score":
                        df.replace(0, np.nan, inplace=True)
                        x = []
                        y = max(df.isnull().sum())
                        z = df.duplicated().sum()
                        box = df.keys()
                        for i in box:
                            if df[i].dtypes != object:
                                x.append(len(df[(np.abs(stats.zscore(df[i])) >= 3)]))
                        error = sum(x)+y+z
                         
                        st.write("Overall, the score of data is ",1-error/len(df))
       
    else:
        st.error("Please select your data to started")

main()