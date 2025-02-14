# conclusion
it seems that the lower the P value the better the performance it's 

but while calculating the correlation should I use log data?
or should I use X - beta* Y to calculate cointegration test?

originally I was just using the raw data :

```python
        bar = {
            self.short_leg: bars[self.short_leg].close,
            self.long_leg: bars[self.long_leg].close,
        }
        self.bar_array.append(bar)
        if len(self.bar_array) < self.array_len:
            return
        p_value = self.coin_test()
```

