"""Train and evaluate from aligned wide CSV: time,AUDUSD,NZDUSD,USDJPY,USDCAD.

No broker account or running MT5 instance is required. Symbols are column names.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from mql5gpu import Library, Model, DEFAULT_TOPOLOGY
from financial_series import read_prices, prepare, score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',required=True)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--symbols',default='AUDUSD,NZDUSD,USDJPY,USDCAD',help='Target first, then predictors')
    parser.add_argument('--sequence',type=int,default=24)
    parser.add_argument('--horizon',type=int,default=6)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--topology',default=DEFAULT_TOPOLOGY)
    parser.add_argument('--output',default='output/model')
    args=parser.parse_args()
    symbols=[s.strip() for s in args.symbols.split(',')]
    if len(symbols)!=len(set(symbols)): parser.error('Duplicate symbols')
    times,prices=read_prices(args.csv,symbols)
    data=prepare(prices,args.sequence,args.horizon)
    directory=Path(args.output); directory.mkdir(parents=True,exist_ok=True)
    with Model(Library(args.dll)) as model:
        model.configure(len(symbols),args.sequence,args.horizon,args.topology)
        training=model.fit(data['train_x'],data['train_y'],epochs=args.epochs)
        metrics=score(model.predict(data['valid_x']),data['valid_y'])
        prediction=model.predict([data['latest']])[0]
        model.save(directory/'weights.txt')
    metadata=dict(schema=1,symbols=symbols,sequence=args.sequence,horizon=args.horizon,topology=args.topology,
        mean=data['mean'],scale=data['scale'],target_scale=100,origin=times[-1].isoformat(),
        validation=metrics,training=training,
        training_last_origin=times[data['train_last_origin']].isoformat(),
        validation_first_origin=times[data['validation_first_origin']].isoformat())
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    result=dict(origin=times[-1].isoformat(),validation=metrics,
        beats_persistence=metrics['ratio'] is not None and metrics['ratio']<1,
        forecast_close=[prices[-1][0]*math.exp(p/100) for p in prediction])
    (directory/'forecast.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
