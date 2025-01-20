import pandas as pd
import numpy as np
import os
from openai import OpenAI
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv('data/FinancialMarketData.csv')
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

# plot out some columns to compare and better understand the data
interesting_cols = ["VIX", "DXY", "MXJP", "XAU BGNL", "JPY"]
plt.figure(figsize=(20, 10))
fig, axes = plt.subplots(1, len(interesting_cols), figsize=(20, 10))

for i, col in enumerate(interesting_cols):
  sns.histplot(
      data=df,
      x=col,
      hue='Y',
      ax=axes[i],
      bins=20
  )
  axes[i].set_title(f'Distribution of {col} by Y')

plt.tight_layout()

st.write("### Distribution of Selected Financial Variables by Y (Anomaly Indicator)")
st.pyplot(fig)

# we can see above that the VIX histogram has the most significant difference so we take that to train the model
# X = df[["VIX", "DXY", "MXJP", "XAU BGNL", "JPY"]]
# y = df ["Y"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().fit_transform(X_test)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# # use logstic regression to train the model
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)

# # evaluate the model
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# precision = all the anomalous days how many did it correctly predict
# recall = how many anomalous days are in all the days --- actually more important than precision in this case


def market_time_split(df, train_years=5, validation_weeks=52, test_weeks=52):
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')

    train_size = train_years * 52
    validation_end = train_size + validation_weeks

    train = df.iloc[:train_size]
    validation = df.iloc[train_size:validation_end]
    test = df.iloc[validation_end:validation_end + test_weeks:]

    print(f'Train size: {len(train)}')
    print(f'Validation size: {len(validation)}')
    print(f'Test size: {len(test)}')

    print(f"Train period: {train['Data'].min()} to {train['Data'].max()}")
    print(f"Train period: {validation['Data'].min()} to {validation['Data'].max()}")
    print(f"Train period: {test['Data'].min()} to {test['Data'].max()}")

    return train, validation, test

train, validation, test = market_time_split(df)

# we want to increase the recall value --> we have to use hyperparameters to tune it
# also we want to now include the time split that we've implemented
X_train = train[["VIX", "DXY", "MXJP", "XAU BGNL", "JPY"]]
y_train = train["Y"]

X_validation = validation[["VIX", "DXY", "MXJP", "XAU BGNL", "JPY"]]
y_validation = validation["Y"]

X_test = test[["VIX", "DXY", "MXJP", "XAU BGNL", "JPY"]]
y_test = test["Y"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", random_state=42, solver="liblinear", max_iter=1500)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_validation_scaled)
print(classification_report(y_validation, y_pred)) 

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")



def get_anomaly_likelihood(recall):
    if recall < 0.30:
        return "not"
    elif recall < 0.65:
        return "somewhat"
    else:
        return "very"



def strategy_suggestion(recall):
    anomaly_likelihood = get_anomaly_likelihood(recall)
    prompt = f'''
        You are an investor concerned about potential economic downturns and want to develop an investment strategy that minimizes risk while maximizing returns based on current market conditions.

        You have trained a machine learning model on historical financial market data to aid your decision, and this model predicts whether the market is likely to experience an anomaly (a significant downturn or fluctuation) in the near future and outputs the following prediction:

        The model predicts that the stock market is {anomaly_likelihood} likely to enter an anomaly in the near future.

        Based on this prediction, you should consider the following investment strategy:

        If the stock market is likely to enter an anomaly (high likelihood), then the strategy is to invest conservatively, focusing on low-risk assets such as bonds and cash. The goal is to minimize potential losses during volatile periods.

        If the stock market is not likely to enter an anomaly (low likelihood), then the strategy is to invest more aggressively in high-risk assets such as equities and stocks, aiming for higher returns when the market is stable.

        If the stock market is somewhat likely to enter an anomaly (moderate likelihood), the strategy should be a balanced approach. Invest in a mix of equities and bonds with diversification, while also keeping a portion in cash for flexibility. This reduces the risk of loss while still positioning for growth.

        This approach is designed to align your investment decisions with the likelihood of market anomalies, helping you navigate different market conditions with a more informed strategy.
        '''

    response = client.chat.completions.create(
        model='llama-3.2-3b-preview',
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return response.choices[0].message.content


explanation = strategy_suggestion(recall)
st.markdown("---")
st.markdown("### Investment Strategy Suggestion")
st.markdown(explanation)