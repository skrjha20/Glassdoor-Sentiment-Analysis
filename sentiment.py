import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

def return_one_year_data(mastek):
    mastek['date'] = pd.to_datetime(mastek.date)
    mastek = mastek.sort_values(by='date', ascending=False)
    mastek = mastek.set_index('date')

    present_date = pd.to_datetime('today').date()
    last_year_date = present_date - pd.DateOffset(years=2)
    data = mastek.loc[present_date: last_year_date.date()]

    data = data.reset_index()
    data = data.drop_duplicates(
        subset=['employee_title', 'location', 'employee_status', 'review_title', 'pros', 'cons'],
        keep='first').reset_index(drop=True)
    return data

def return_sentiments(data):
    data =  data[data['review_title'].notnull()].reset_index(drop=True)
    data = data[data['review_title'] != '-'].reset_index(drop=True)

    data['sentiment'] = [None] * len(data)
    sia = SentimentIntensityAnalyzer()
    for i in range(len(data)):
        try:
            senti = sia.polarity_scores(data['review_title'][i])
            if senti['neg'] > senti['pos'] and senti['neg'] > senti['neu']:
                data['sentiment'][i] = 'Negative'
            elif senti['neu'] > senti['pos'] and senti['neu'] > senti['neg']:
                data['sentiment'][i] = 'Neutral'
            elif senti['pos'] > senti['neg'] and senti['pos'] > senti['neu']:
                data['sentiment'][i] = 'Positive'
            else:
                print(i)
        except:
            data['sentiment'][i] = 'NaN'

    return data

def data_analytics_location(data):
    data['location'] = data['location'].replace(['Gurgaon, Haryana, Haryana', 'Greater Noida', 'New Delhi'], 'NCR')
    data['location'] = data['location'].replace(['Andheri', 'Andheri East', 'Ghansoli', 'Mahape', 'Navi Mumbai'],
                                                'Mumbai')
    data['location'] = data['location'].replace('New York, NY', 'New York')
    data['location'] = data['location'].fillna('Anonymous')

    df = data.groupby(['location', 'sentiment']).sum()
    df = df.drop(['helpful', 'rating_overall', 'rating_balance', 'rating_culture', 'rating_career', 'rating_comp',
                  'rating_mgmt'], axis=1)
    df.to_csv('data_location.csv')

def data_analytics_employee_title(data):
    data['employee_title'] = data['employee_title'].str.lstrip()
    data['employee_title'] = data['employee_title'].str.rstrip()
    data['employee_title'] = data['employee_title'].replace(['Junior Software Developer', 'Software Engineer',
                                                             'Software Developer', 'Trainee Software Engineer',
                                                             'Developmental Engineer', 'Executive'],
                                                            'Software Engineer')
    data['employee_title'] = data['employee_title'].replace(['Senior UI Developer', 'Senior Software Engineer',
                                                             'Project Engineer', 'BI Developer', 'SQL',
                                                             'Senior Associate', 'RPA Developer Uipath', 'IOS Engineer',
                                                             'SQL Server Database Administrator','ASP.NET Developer',
                                                             'Senior Software Developer', 'S S E(Senior Software Engineer)',
                                                             'Senor Software Engineer'],
                                                            'Senior Software Engineer')
    data['employee_title'] = data['employee_title'].replace(['Software Test Engineer', 'Test Analyst',
                                                             'SeniorTest Engineer', 'Business Analyst',
                                                             'Senior Business Analyst', 'Test Engineer',
                                                             'Senior Test Engineer', 'QA Test Lead',
                                                             'Automation Test Engineer','PMO'],
                                                            'QA Role')
    data['employee_title'] = data['employee_title'].replace(['Software Specialist', 'Technical Lead',
                                                             'Associate Solution Architect', 'Solutions Architect',
                                                             'Scrum Master', 'Associate Business Consultant',
                                                             'Data Architect', 'Solution Architect', 'Team Leader',
                                                             'Technical Analyst'],
                                                            'Solution Architect')
    data['employee_title'] = data['employee_title'].replace(['Program Manager', 'Manager FP&A', 'Senior Project Leader',
                                                             'AVP', 'Project Manager', 'Account Executive',
                                                             'Group Manager','Senior Executive Marketing', 'Founder',
                                                             'Senior Sales Executive', 'Project Leader',
                                                             'Human Resources Assistant Manager', 'Manager'],
                                                            'Manager')

    df = data.groupby(['employee_title', 'sentiment']).sum()
    df = df.drop(['helpful', 'rating_overall', 'rating_balance', 'rating_culture', 'rating_career', 'rating_comp',
                  'rating_mgmt'], axis=1)
    df.to_csv('data_employee.csv')
    return data

def data_analytics_ratings(data):
    df = data.groupby(['employee_title']).mean()
    df = df.drop(['helpful', 'ones'], axis=1)
    df['rating_overall'] = np.round(df['rating_overall'], 2)
    df['rating_balance'] = np.round(df['rating_balance'], 2)
    df['rating_culture'] = np.round(df['rating_culture'], 2)
    df['rating_career'] = np.round(df['rating_career'], 2)
    df['rating_comp'] = np.round(df['rating_comp'], 2)
    df['rating_mgmt'] = np.round(df['rating_mgmt'], 2)
    df = df.reset_index()
    ax = sns.barplot(x="employee_title", y="rating_overall", hue="rating_comp", data=df)
    ax.show()
    df.to_csv('data_ratings.csv')


if __name__ == '__main__':
    mastek = pd.read_csv('mastek.csv')
    data = return_one_year_data(mastek)
    data = return_sentiments(data)

    data['ones'] = 1
    data_analytics_location(data)
    data = data_analytics_employee_title(data)
    data_analytics_ratings(data)
    #data.to_csv("without_dup_data.csv", index=False)
