"""Inference from an existing model bundle without MT5 or retraining."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from mql5gpu import Library,Model
from financial_series import read_prices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',required=True); parser.add_argument('--csv',required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    folder=Path(args.model)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata['schema']!=1: raise ValueError('Unsupported metadata schema')
    times,prices=read_prices(args.csv,metadata['symbols'])
    sequence=metadata['sequence']; features=len(metadata['symbols'])
    if len(prices)<sequence+1: raise ValueError('Insufficient recent rows')
    row=[]
    for i in range(len(prices)-sequence,len(prices)):
        for f in range(features):
            scale=metadata['scale'][f]
            if not math.isfinite(scale) or scale<=0: raise ValueError('Invalid stored scaler')
            row.append((math.log(prices[i][f]/prices[i-1][f])-metadata['mean'][f])/scale)
    with Model(Library(args.dll)) as model:
        model.load(folder/'weights.txt')
        if model.dimensions!=(features,sequence,metadata['horizon']): raise ValueError('Metadata/model shape mismatch')
        predicted=model.predict([row])[0]
    print(json.dumps(dict(origin=times[-1].isoformat(),forecast_close=[prices[-1][0]*math.exp(v/metadata['target_scale']) for v in predicted]),indent=2,allow_nan=False))


if __name__=='__main__': main()
