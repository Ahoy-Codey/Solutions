import numpy as np

def create_and_analyze_student_data():

    np.random.seed(42)
    names = ['Tony Stark','Peter Parker','MJ','Natasha','Thor','Loki','Steve Rogers','Strange','Wanda','Peter-Quill']
    subjects = np.random.choice(['Math', 'Science','Art'], 10)
    scores = np.random.randint(50, 101, 10)

    df = pd.DataFrame({
        'Name': names,
        'Subject': subjects,
        'Score': scores,
        'Grade': ''
    })
    print("DataFrame with No Grades Assigned:")
    print(df,'\n')

    def assign_grade(score):
        if 90 <= score <= 100:
            return 'A'
        elif 80 <= score <= 89:
            return 'B'
        elif 70 <= score <= 79:
            return 'C'
        elif 60 <= score <= 69:
            return 'D'
        else:
            return 'F'

    df['Grade'] = df['Score'].apply(assign_grade)
    print("DataFrame with Grades Assigned:")
    print(df,'\n')

    df_sorted_by_score = df.sort_values(by='Score', ascending=False)
    print("DataFrame Sorted by Score (Descending):")
    print(df_sorted_by_score,'\n')

    average_scores_per_subject = df.groupby('Subject')['Score'].mean()
    print("Average Score for Each Subject:")
    print(average_scores_per_subject,'\n')

    def pandas_filter_pass(dataframe):
        return dataframe[dataframe['Grade'].isin(['A', 'B'])].copy()

    passing_students_df = pandas_filter_pass(df)
    print("DataFrame with only Passing Students (Grades A or B):")
    print(passing_students_df,'\n')

create_and_analyze_student_data()
