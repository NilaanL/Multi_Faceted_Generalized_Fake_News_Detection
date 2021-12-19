def normalize(dataFrame,features , parameterDict={}): 
    """funtion normalize given dataFrame with Min-Max normalization and Z-score  ;

    Args:
        dataFrame (Pandas.Dataframe): dataframe to be processsed
        features (List<String>): feature to apply normalization
        parameterDict (dict, optional): normalization parameter Dict. Defaults to {}.

    Returns:
        Pandas.Dataframe: normalized dataframe
    """
    dataframe=dataFrame.copy()
    for column in dataframe[features].columns.tolist():
        Q1=dataframe[column].quantile(0.25)
        Q3=dataframe[column].quantile(0.75)
        # Q1=parameterDict[column]["Q1"]
        # Q3=parameterDict[column]["Q3"]
        IQR=(Q3-Q1)
        minV=Q1 - 1.5*IQR
        maxV=Q3 + 1.5*IQR

        # IQR=parameterDict[column]["IQR"]
        # minV=parameterDict[column]["minV"]
        # maxV=parameterDict[column]["maxV"]

        temp=dataframe[column].copy()

        dataframe[column]=dataframe[column].apply(lambda x:minV if x< minV else maxV if x>maxV else x)
        mean = dataframe[column].mean()
        std  = dataframe[column].std()
        # mean=parameterDict[column]["mean"]
        # std=parameterDict[column]["std"]

        dataframe[column]=dataframe[column].apply(lambda x: 0 if (std==0) else (x-mean)/std)
        if (dataframe[column].sum()==0 and dataframe[column].std()==0):
            dataframe[column]=temp.apply(lambda x : 1 if x>0 else 0)
    return dataframe

