import Investment_backtest as imbt
var_calc=imbt.Varible_Calculate(begin='2000-01',end='2020-12',path_ori='测试数据')
# 数据频率
freq='day' # 或着'day'
# 基础数据（月）
ret=var_calc.calc_ret(freq=freq)
size=var_calc.calc_size()
# 剔除投资域数据
Listshare=var_calc.calc_Listshare(exMonth=12)
YMshare=var_calc.calc_YMshare(min_month=15,min_year=120)
STshare=var_calc.calc_STshare()
exclude=Listshare*YMshare*STshare
# 构造策略
strategy=imbt.Strategy(exclude,size)
strategy.load_factor(factor_list=[size],factor_name=['factor'])
strategy.fit(ret,type=freq)
portfolio,group_tickers=strategy.get_portfolio_ret(group_name=['size30','size70'])
# 策略展示
performance=imbt.Performance(portfolio,path_ori='测试数据',freq=freq)
perf=performance.performance_valuation()
performance.plot_exNetValue()