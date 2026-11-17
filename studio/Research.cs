using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Web.Script.Serialization;

namespace Remind.DNForecast.Studio
{
    internal static class Research
    {
        private static double Number(string value)
        {
            double n=double.Parse(value,NumberStyles.Float,CultureInfo.InvariantCulture);
            if(double.IsNaN(n) || double.IsInfinity(n)) throw new InvalidDataException("Invalid value in forecast trace");
            return n;
        }

        // Chart points are keyed to the outcome time. Decisions use the previous origin row.
        internal static void Evaluate(string trace,string output,Recipe config)
        {
            var time=new List<long>(); var price=new List<double>(); var prediction=new List<double[]>();
            foreach(string line in File.ReadLines(trace).Skip(1)) {
                string[] fields=line.Split(',');
                if(fields.Length!=6) throw new InvalidDataException("Invalid forecast trace");
                time.Add(long.Parse(fields[0],CultureInfo.InvariantCulture)); price.Add(Number(fields[1]));
                prediction.Add(fields[2].Length==0 ? null : fields.Skip(2).Select(Number).ToArray());
            }
            int h=config.horizons;
            double equity=1,peak=1,drawdown=0,units=0,cash=1,threshold=config.edge/10000.0;
            double fee=config.cost/10000.0, fraction=config.allocation/100.0;
            int exit=-1,wins=0,exposed=0,entry=0; double initialEquity=1;
            var points=new List<object>(); var trades=new List<object>();
            for(int i=0;i<time.Count;i++) {
                if(units!=0) exposed++;
                if(units!=0 && (exit==i || cash+units*price[i]<=0)) {
                    cash+=units*price[i]-Math.Abs(units)*price[i]*fee;
                    cash=Math.Max(0,cash);
                    double gain=cash/initialEquity-1;
                    trades.Add(new { signal=time[entry-1],entry=time[entry],exit=time[i],side=units>0?"long":"short",gain=gain });
                    if(gain>0) wins++;
                    units=0;
                }
                // Signal known at i-1, entry at i, exit at i+h. No overlapping positions.
                if(units==0 && cash>0 && i>0 && i+h<time.Count && prediction[i-1]!=null) {
                    double[] forecast=prediction[i-1];
                    int side=forecast[0]>threshold && config.side!="short" ? 1 :
                        forecast[0]<-threshold && config.side!="long" ? -1 : 0;
                    if(side!=0) {
                        initialEquity=cash;entry=i;exit=i+h;
                        units=side*cash*fraction/price[i];
                        cash-=units*price[i]+Math.Abs(units)*price[i]*fee;
                    }
                }
                equity=Math.Max(0,cash+units*price[i]);
                peak=Math.Max(peak,equity);drawdown=Math.Max(drawdown,1-equity/peak);
                double[] p=i>=h ? prediction[i-h] : null;
                var row=new double?[]{time[i],price[i],null,null,null,equity};
                if(p!=null) for(int k=0;k<3;k++) row[k+2]=price[i-h]*Math.Exp(p[k]);
                points.Add(row);
            }
            var json=new JavaScriptSerializer { MaxJsonLength=16000000 };
            File.WriteAllText(output,json.Serialize(new { horizon=h,points=points,trades=trades,net=equity-1,
                drawdown=drawdown,winRate=trades.Count>0 ? (double?)wins/trades.Count : null,
                exposure=(double)exposed/Math.Max(1,time.Count-1),config=config }));
        }
    }
}
