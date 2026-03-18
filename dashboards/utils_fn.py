CAT_LIST = ['Sex', 'Embarked']
 
def embarked_imputation(df):
    df = df.copy()
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    return df
 
def age_imputation(df):
    df = df.copy()
    age_median = df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)
    return df
 
def feature_engineer(df):
    df = df.copy()
    df['familySize'] = df['SibSp'] + df['Parch'] + 1
    return df
 
def drop_features(df):
    cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    cols_to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_to_drop)
 
def as_category(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype('category')
    return df
 
def full_preprocess(raw_df):
    """Run the exact same pipeline used during training."""
    df = embarked_imputation(raw_df)
    df = age_imputation(df)
    df = feature_engineer(df)
    df = drop_features(df)
    if 'Survived' in df.columns:
        df = df.drop(columns=['Survived'])
    # fill any remaining Fare NaN
    if df['Fare'].isna().any():
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = as_category(df, CAT_LIST)
    return df