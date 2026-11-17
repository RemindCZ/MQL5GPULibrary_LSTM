"""Fit a direct multi-horizon challenger and export a self-contained ONNX artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from dnforecast import Library, Session, market_contract
from forecast_series import losses, read_bars, window


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = nn.ConstantPad1d((2*dilation,0),0.)
        self.conv = nn.Conv1d(width,width,3,dilation=dilation)

    def forward(self,x):
        return x + F.relu(self.conv(self.pad(x)))


class Network(nn.Module):
    def __init__(self, kind, channels, horizons, mean, scale, target_scale):
        super().__init__()
        self.kind=kind
        self.h=horizons
        for key,val in (("mean",mean),("scale",scale),("target_scale",target_scale)):
            self.register_buffer(key,torch.as_tensor(val,dtype=torch.float32))
        self.register_buffer("calibration",torch.zeros(horizons,3))
        width=16
        if kind=="tcn":
            self.body=nn.Sequential(nn.Conv1d(channels,width,1),*(CausalBlock(width,d) for d in (1,2,4,8,16,32)))
        elif kind=="gru":
            self.body=nn.GRU(channels,width,batch_first=True)
        else:
            width=channels*8
        self.head=nn.Linear(width,2*horizons)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def estimates(self,x):
        z=(x-self.mean)/self.scale
        if self.kind=="tcn":
            z=self.body(z.transpose(1,2))[:,:,-1]
        elif self.kind=="gru":
            z=self.body(z)[0][:,-1,:]
        else:
            z=z[:,-8:,:].flatten(1)
        mean,logit=self.head(z).split(self.h,dim=1)
        return mean,logit

    def volatility(self,x):
        v=x[:,-16:,0].square().mean(dim=1,keepdim=True).clamp_min(1e-16)
        return v*torch.arange(1,self.h+1,dtype=x.dtype,device=x.device)

    def forward(self,x):
        mean,logit=self.estimates(x)
        point=mean*self.target_scale
        variance=self.volatility(x)
        bands=point.unsqueeze(-1)+variance.sqrt().unsqueeze(-1)*self.calibration
        return torch.cat((point.unsqueeze(-1),bands,logit.sigmoid().unsqueeze(-1),variance.unsqueeze(-1)),dim=2)


def run(args):
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Choose a new model filename to preserve existing versions")
    symbols=args.symbols.split(",")
    contract=market_contract(symbols,args.seconds)
    bars=read_bars(args.bars,symbols)
    s,h=args.steps,args.horizons
    if not (16<=s<=1024 and 1<=h<=min(64,s//2) and args.epochs>=1):
        raise ValueError("Invalid model shape or epoch count")
    lib=Library(args.dll)
    cut1,cut2,cut3=(int(len(bars)*v) for v in (.55,.70,.85))
    train=np.arange(s,cut1-h)
    selection=np.arange(cut1,cut2-h)
    calibration=np.arange(cut2,cut3-h)
    test=np.arange(cut3,len(bars)-h)
    blocks=(train,selection,calibration,test)
    if min(map(len,blocks))<32:raise ValueError("Not enough observations in each purged block")
    for block,next_start in zip(blocks,(cut1,cut2,cut3)):
        last=int(block[-1])+h
        if bars[last].available>=bars[next_start].close_time:
            raise ValueError("A label was still unavailable at the next chronological split")
    x=np.empty((len(bars)-h,s,3*len(symbols)),np.float32)
    y=np.empty((len(x),h),np.float32)
    for i in range(s,len(x)):
        x[i]=np.ctypeslib.as_array(lib.encode(s,len(symbols),window(bars,i,s))).reshape(s,-1)
        y[i]=np.log([b.prices[0]/bars[i].prices[0] for b in bars[i+1:i+h+1]])
    # Unique final steps of training windows suffice for feature scaling.
    sample=x[train,-1,:]
    mean=sample.mean(axis=0)
    scale=np.maximum(sample.std(axis=0),1e-6)
    target_scale=np.maximum(y[train].std(axis=0),1e-8)
    x,y=torch.from_numpy(x),torch.from_numpy(y)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model=Network(args.kind,x.shape[2],h,mean,scale,target_scale)
    if args.kind=="ridge":
        z=((x[train,-8:]-model.mean)/model.scale).flatten(1).double()
        target=(y[train]/model.target_scale).double()
        zm,tm=z.mean(0),target.mean(0)
        z=z-zm
        weights=torch.linalg.solve(z.T@z+len(train)*torch.eye(z.shape[1],dtype=z.dtype),z.T@(target-tm))
        with torch.no_grad():
            model.head.weight[:h].copy_(weights.T)
            model.head.bias[:h].copy_(tm-zm@weights)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        best=float("inf")
        for _ in range(args.epochs):
            for ids in torch.as_tensor(train)[torch.randperm(len(train))].split(128):
                point,logits=model.estimates(x[ids])
                loss=F.mse_loss(point,y[ids]/model.target_scale)+F.binary_cross_entropy_with_logits(logits,(y[ids]>0).float())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            with torch.no_grad():
                errors=[(model.estimates(x[b])[0]-y[b]/model.target_scale).square().sum()
                        for b in torch.as_tensor(selection).split(128)]
                score=sum(v.item() for v in errors)/(len(selection)*h)
                if score<best:
                    best=score
                    selected={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(selected)
    model.eval()
    with torch.no_grad():
        residual=[]
        for b in torch.as_tensor(calibration).split(128):
            result=model(x[b])
            residual.append((y[b]-result[:,:,0])/result[:,:,5].sqrt())
        model.calibration.copy_(torch.quantile(torch.cat(residual),torch.tensor([.1,.5,.9]),dim=0).T)
        zero=torch.zeros(1,s,x.shape[2])
        expected=model(zero).flatten().tolist()
    import tempfile
    # Validate the exact serialized graph with the production DLL before publishing it.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        path=Path(tmp)/"model.onnx"
        torch.onnx.export(model,(zero,),str(path),input_names=["features"],output_names=["forecast"],
                          opset_version=17,dynamo=False)
        graph=onnx.load(path)
        onnx.helper.set_model_props(graph,{"dn3.contract":contract,"dn3.zero_probe":" ".join(map(str,expected)),
                                           "dn3.kind":args.kind,"dn3.status":"research"})
        onnx.checker.check_model(graph)
        onnx.save(graph,path)
        scores=np.zeros((2,h,6))
        max_diff=0.
        with Session(lib,symbols,args.seconds,s,h,model=path) as candidate, Session(lib,symbols,args.seconds,s,h) as ref:
            for index,i in enumerate(test):
                prices=window(bars,i,s)
                outputs=[session.predict(bars[i].close_time,prices) for session in (candidate,ref)]
                if index<16:
                    with torch.no_grad(): native=model(x[i:i+1]).numpy()[0]
                    np.testing.assert_allclose(outputs[0],native,rtol=2e-4,atol=1e-8)
                    max_diff=max(max_diff,float(np.max(np.abs(np.asarray(outputs[0])-native))))
                rv=0.
                for j in range(h):
                    rv+=np.log(bars[i+j+1].prices[0]/bars[i+j].prices[0])**2
                    for k in (0,1):
                        scores[k,j]+=losses(outputs[k][j],float(y[i,j]),rv)
        scores/=len(test)
        report=dict(contract=contract,kind=args.kind,status="research",seed=args.seed,steps=s,horizons=h,
                    data_sha256=hashlib.sha256(args.bars.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parity_max_abs=max_diff,
                    blocks={name:dict(count=len(b),first=bars[b[0]].close_time,last=bars[b[-1]].close_time)
                            for name,b in zip(("train","selection","calibration","test"),blocks)},
                    metrics=["mse","brier","pinball","qlike","coverage","width"],
                    candidate=scores[0].tolist(),reference=scores[1].tolist())
        # Write a new file under the destination ACL, not the private temp directory ACL.
        with args.output.open("xb") as final:
            final.write(path.read_bytes())
        args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(args.output,"test origins",len(test),"point MSE ratios",scores[0,:,0]/np.maximum(scores[1,:,0],1e-30))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dll","bars","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--symbols",required=True)
    p.add_argument("--seconds",type=int,default=3600)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--horizons",type=int,default=6)
    p.add_argument("--kind",choices=("ridge","gru","tcn"),default="tcn")
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--seed",type=int,default=37)
    run(p.parse_args())
