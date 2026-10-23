import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from financial_series import prepare, score


class PreparationTests(unittest.TestCase):
    def test_purge_and_training_only_scaler(self):
        prices=[[math.exp(.0002*i+.001*math.sin(i)),math.exp(.0001*i)] for i in range(500)]
        a=prepare(prices,12,6)
        self.assertLess(a['train_last_origin']+6,a['validation_first_origin'])
        altered=[row.copy() for row in prices]
        for i in range(a['validation_first_origin'],len(altered)): altered[i][0]*=10
        b=prepare(altered,12,6)
        self.assertEqual(a['mean'],b['mean']); self.assertEqual(a['scale'],b['scale'])
        self.assertEqual(a['train_x'],b['train_x']); self.assertEqual(a['train_y'],b['train_y'])
        self.assertEqual(len(a['latest']),24)

    def test_baseline_and_invalid_data(self):
        result=score([[0,0],[0,0]],[[.1,.2],[-.1,.3]])
        self.assertEqual(result['ratio'],1)
        self.assertEqual(score([[1,2]],[[1,2]])['mse'],0)
        with self.assertRaises(ValueError): prepare([[1],[float('nan')]],12,6)
        with self.assertRaises(ValueError): prepare([[1]]*20,12,6)
        with self.assertRaises(ValueError): score([[]],[[]])
        with self.assertRaises(ValueError): score([[1],[1,2]],[[1],[1,2]])


if __name__=='__main__': unittest.main()
