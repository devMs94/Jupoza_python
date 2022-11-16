# for production use
# export FLASK_ENV=development
###  Risk parity  ###
import numpy as np
import scipy.optimize as sco
import pandas as pd

from flask import Flask, request, jsonify


# 서버 구현을 위한 Flask 객체 import

def Risk_Contribution(weight):
    global covmat
    weight = np.array(weight)
    std = np.sqrt(np.dot(weight.T, np.dot(covmat, weight)))
    mrc = np.dot(covmat, weight) / std
    rc = weight * mrc

    return rc, std


def risk_parity_optimization(df):
    global covmat
    TOLERANCE = 1e-20
    num_assets = len(covmat)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
    result = sco.minimize(risk_parity_target, num_assets * [1. / num_assets, ], method='SLSQP',
                          constraints=constraints, tol=TOLERANCE)
    Risk_Parity_Allocation = pd.DataFrame(result.x, index=df.columns, columns=['allocation'])

    return round(Risk_Parity_Allocation * 100, 2)


def risk_parity_target(weight):
    rc, std = Risk_Contribution(weight)
    RC_assets = rc
    RC_target = std / len(rc)
    objective_fun = np.sum(np.square(RC_assets - RC_target.T))

    return objective_fun


def make_df(adj_price, stock):
    universe = adj_price[stock].loc['2016-01-01':'2021-12-31']
    df = universe.pct_change(1)
    df = df[1:]
    df = df.dropna(axis='index')
    return df


def RP(adj_price, stock):
    df = make_df(adj_price, stock)
    rpo = risk_parity_optimization(df)
    return rpo


app = Flask(__name__)

DATA_FILE = "./KRX100재무주가.xlsx"
data_wb = pd.ExcelFile(DATA_FILE)
adj_price_ts = data_wb.parse("주가", header=[0, 1, 2], index_col=0)
adj_price = adj_price_ts.xs('수정주가(원)', axis=1, level=2)
adj_price.columns = adj_price.columns.droplevel(0)
covmat = np.array([])


@app.route('/', methods=['POST'])
def root():
    param = request.get_json()
    stockList = []
    if "weightRequest" in param:
        stockList = param["weightRequest"]
    else:
        return jsonify({
            "result": "failure"
        })
    if len(stockList) != 5:
        return jsonify({
            "result": "failure"
        })
    global covmat
    df = make_df(adj_price, stockList)
    covmat = np.array(df.cov() * 240)
    rp = np.array(RP(adj_price, stockList))
    ret = []
    for a in rp:
        ret.append(a[0])
    return jsonify({
        "result": "success",
        "weightResponse": ret
    })


if __name__ == "__main__":
    app.run()

