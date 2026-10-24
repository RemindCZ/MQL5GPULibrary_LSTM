"""Chronological financial-series preparation independent of CUDA/MT5."""
import csv
from datetime import datetime
import math


def read_prices(path, symbols):
    if not symbols or len(set(symbols)) != len(symbols) or any(not symbol for symbol in symbols):
        raise ValueError('Choose unique nonempty symbol names, with the target first')
    times, prices = [], []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('CSV timestamps must include an explicit UTC offset')
            if times and stamp <= times[-1]:
                raise ValueError('Timestamps must be strictly increasing and unique')
            values = [float(row[s]) for s in symbols]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('All aligned close prices must be positive and finite')
            times.append(stamp); prices.append(values)
    return times, prices


def prepare(prices, sequence, horizon, validation=.2):
    if sequence < 2 or horizon < 1 or not .1 <= validation <= .4 or not prices:
        raise ValueError('Invalid sequence, horizon, validation fraction or data')
    features = len(prices[0])
    if not features or any(len(p) != features or any(not math.isfinite(v) or v <= 0 for v in p) for p in prices):
        raise ValueError('Prices must be a rectangular array of positive finite values')
    returns = [[0.0]*features] + [[math.log(b/a) for a,b in zip(prices[i-1],prices[i])] for i in range(1,len(prices))]
    last = len(prices)-1-horizon
    cut = sequence + int((last-sequence+1)*(1-validation))
    train_end = cut-horizon-1
    if train_end < sequence+64 or last-cut < 32:
        raise ValueError('Insufficient data for a purged chronological holdout')
    fit_rows = returns[1:train_end+1]
    mean = [sum(row[f] for row in fit_rows)/len(fit_rows) for f in range(features)]
    scale = [max(1e-8,math.sqrt(sum((row[f]-mean[f])**2 for row in fit_rows)/len(fit_rows))) for f in range(features)]
    def encode(end):
        return [(returns[t][f]-mean[f])/scale[f] for t in range(end-sequence+1,end+1) for f in range(features)]
    def target(end): return [100*math.log(prices[end+h][0]/prices[end][0]) for h in range(1,horizon+1)]
    train = list(range(sequence,train_end+1)); valid = list(range(cut,last+1))
    return dict(train_x=[encode(i) for i in train], train_y=[target(i) for i in train],
                valid_x=[encode(i) for i in valid], valid_y=[target(i) for i in valid], latest=encode(len(prices)-1),
                mean=mean, scale=scale, train_last_origin=train_end, validation_first_origin=cut,
                sequence=sequence, horizon=horizon, features=features)


def score(predicted, targets):
    if not targets or not targets[0] or len(predicted)!=len(targets) or any(
        len(a)!=len(targets[0]) or len(b)!=len(targets[0]) for a,b in zip(predicted,targets)):
        raise ValueError('Prediction/target shapes differ')
    if any(not math.isfinite(v) for row in predicted+targets for v in row):
        raise ValueError('Non-finite prediction or target')
    horizon=len(targets[0]); n=len(targets)
    residual=[sum((a[h]-b[h])**2 for a,b in zip(predicted,targets))/n for h in range(horizon)]
    mse=sum(residual)/horizon
    baseline=sum(v*v for row in targets for v in row)/(n*horizon)
    return dict(mse=mse, baseline_mse=baseline, ratio=mse/baseline if baseline>1e-20 else None,
                residual_rmse=[math.sqrt(v)/100 for v in residual])
