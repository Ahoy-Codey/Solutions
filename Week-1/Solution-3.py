import numpy as np
import pandas as pd

def grades(marks):
    if(marks<60):
        return 'F'
    elif(marks<70):
        return 'D'
    elif(marks<80):
        return 'C'
    elif(marks<90):
        return 'B'
    elif(marks<=100):
        return 'A'

def pandas_filter_pass(dataframe):
    return dataframe[(dataframe['Grade']=='A') | (dataframe['Grade']=='B')]

df = pd.DataFrame({
    'Name': ['James', 'Emily', 'Michael', 'Sarah', 'John', 'Ashley', 'David', 'Jessica', 'Daniel', 'Amanda'],
    'Subject': ['Math', 'Science', 'English', 'History', 'Geography', 'Physics', 'Chemistry', 'Biology', 'Economics', 'Computer'],
    'Score': np.random.randint(50, 101, size=10),
    'Grade': [''] * 10
})

df['Grade']=df['Score'].map(grades)

df=df.sort_values('Score')
print(df)
am=df['Score'].mean()
print('Mean is',am)

print(pandas_filter_pass(df))