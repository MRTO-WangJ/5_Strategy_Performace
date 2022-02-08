import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
# 类库包括三个部分：0、数据接口层；1、数据计算层；2、投资策略和回测计算部分（Strategy）；3、图片列表展示部分（Performance）
# 第一部分设定总投资域和样本时间，包括特殊剔除样本，如上市不满一年、ST股票等。
# 第二部分主要是投资策略编写以及收益序列的计算，包括因子投资、交叉线策略或着机器学习等。
# 第三部分主要根据投资策略收益时间序列计算组合收益评价指标，并且用图表方式展现。

# 类支持月度调仓
# 类库基于日度or月度数据

#################################################################################################################################
# 底层数据适配层
# 数据字段名称适配：1、股票代码统一用ticker表示；2、日度时间戳用tradeDate表示；3、月度时间戳用month表示；4、收益率用ret表示；5、其余用原数据字段名表示。
# 数据获取一定需要字段：1、股票代码ticker；2、交易时间tradeDate或着month
#################################################################################################################################
class Csmar_Api():
    # 数据文件路径
    def __init__(self,path_ori):
        self.path_ori=path_ori

    # 月度交易数据表格适配
    def api_TRD_Mnth(self,filename, col=[]):
        path=os.path.join(self.path_ori,filename)
        if len(col) == 0:
            data = pd.read_csv(path)
        else:
            data = pd.read_csv(path, usecols=col)
        data = data.rename(columns={'Stkcd': 'ticker', 'Trdmnt': 'month'})
        data['ticker'] = data['ticker'].astype(int).astype(str).str.zfill(6)
        return data

    # 日度交易数据表格适配
    def api_TRD_Dalyr(self,filename, col=[]):
        path = os.path.join(self.path_ori, filename)
        if len(col) == 0:
            data = pd.read_csv(path)
        else:
            data = pd.read_csv(path, usecols=col)
        data = data.rename(columns={'Stkcd': 'ticker', 'Trddt': 'tradeDate'})
        data['ticker'] = data['ticker'].astype(int).astype(str).str.zfill(6)
        return data

    # 公司文件表格适配
    def api_TRD_Co(self,filename, col=[]):
        path = os.path.join(self.path_ori, filename)
        if len(col) == 0:
            data = pd.read_csv(path)
        else:
            data = pd.read_csv(path, usecols=col)
        data = data.rename(columns={'Stkcd': 'ticker'})
        data['ticker'] = data['ticker'].astype(int).astype(str).str.zfill(6)
        return data

    # 月度FF因子表适配
    def api_STK_MKT_FIVEFACMONTH(self,filename,col=[]):
        path = os.path.join(self.path_ori, filename)
        # col=['MarkettypeID','Portfolios','TradingMonth', 'RiskPremium1', 'SMB1','HML1', 'RMW1', 'CMA1']
        if len(col)==0:
            data=pd.read_csv(path)
        else:
            data = pd.read_csv(path, usecols=col)
        data = data[data['MarkettypeID'] == 'P9714']  # 综合A股、创业板、科创板
        data = data[data['Portfolios'] == 1]  # 2*3投资组合划分方法
        data = data[['TradingMonth', 'RiskPremium1', 'SMB1',
                   'HML1', 'RMW1', 'CMA1']]
        data.columns = ['month', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma']
        return data

    # 日度FF因子适配表
    def api_STK_MKT_FIVEFACDAY(self,filename,col=[]):
        path = os.path.join(self.path_ori, filename)
        # col=['MarkettypeID','Portfolios','TradingDate', 'RiskPremium1', 'SMB1','HML1', 'RMW1', 'CMA1']
        if len(col)==0:
            data=pd.read_csv(path)
        else:
            data = pd.read_csv(path, usecols=col)
        data = data[data['MarkettypeID'] == 'P9714']  # 综合A股、创业板、科创板
        data = data[data['Portfolios'] == 1]  # 2*3投资组合划分方法
        data = data[['TradingDate', 'RiskPremium1', 'SMB1',
                   'HML1', 'RMW1', 'CMA1']]
        data.columns = ['tradeDate', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma']
        return data

#################################################################################################################################
# 变量计算层
# 计算加权超额收益需要基础数据：1、日/月度交易收益率；2、月度股票流通市值或着总市值；3、日度无风险收益率；
# 计算股票池剔除数据需要基础数据：1、月度交易数据，月交易天数；2、日度交易数据，交易状态；3、公司信息，上市时间；
# 数据整理思路：在策略部分所有传入数据均为宽列表形式，所以在这一部分投资域计算时均返回宽列表。
# 除收益变量外，其他变量需要滞后
#################################################################################################################################
class Varible_Calculate():
    def __init__(self,begin,end,path_ori):
        self.begin=begin
        self.end=end
        # 设置调用接口对象
        self._csmAPI = Csmar_Api(path_ori=path_ori)

    # 计算收益率
    def calc_ret(self,freq='month'):
        # 收益测试数据格式对齐
        if freq=='month':
            ret_data = self._csmAPI.api_TRD_Mnth(filename='TRD_Mnth.csv', col=['Stkcd', 'Trdmnt', 'Mretwd'])
            ret_data['ret'] = ret_data['Mretwd'] * 100
            ret = ret_data.pivot(index='month', columns='ticker', values='ret')
            month_list=list(ret.index)
        else:
            print('获取日度数据需要一点时间……')
            ret_data=self._csmAPI.api_TRD_Dalyr(filename='TRD_Dalyr.csv',col=['Stkcd','Trddt','Dretwd'])
            ret_data['ret'] = ret_data['Dretwd'] * 100
            ret_data['month']=ret_data['tradeDate'].str[:7]
            month_list=ret_data['month'].sort_values().unique().tolist()
            ret=ret_data.pivot(index='tradeDate', columns='ticker', values='ret')
        # 设置剔除demo：与ret对齐
        self.Demo = pd.DataFrame(index=month_list, columns=ret.columns).reset_index().rename(columns={'index': 'month'})
        self.Demo = self.Demo.melt(id_vars='month', var_name='ticker', value_name='none')
        ret = ret[(ret.index >= self.begin) & (ret.index <= self.end)]
        return ret

    # 计算月度流通市值
    def calc_size(self,size_type='total'):
        size = self._csmAPI.api_TRD_Mnth(filename='TRD_Mnth.csv', col=['Stkcd', 'Trdmnt', 'Msmvosd','Msmvttl'])
        if size_type=='total':
            size=size[['ticker','month','Msmvttl']]
        else:
            size = size[['ticker', 'month', 'Msmvosd']]
        size.columns=['ticker', 'month', 'size']
        self.Demo = self.Demo.merge(size, how='left', on=['ticker', 'month'])
        size=self.Demo.pivot(index='month',columns='ticker',values='size')
        size=size.shift(1)
        size = size[(size.index >= self.begin) & (size.index <= self.end)]
        return size

    # 计算剔除未上市满股票
    def calc_Listshare(self,exMonth=12):
        firm_data = self._csmAPI.api_TRD_Co(filename='TRD_Co.csv', col=['Stkcd', 'Listdt'])
        firm_data['bound'] = (pd.to_datetime(firm_data['Listdt']) + pd.Timedelta(days=30 * (exMonth + 1))).astype(
            str).str[:7]
        self.Demo = self.Demo.merge(firm_data[['ticker', 'bound']], how='left', on='ticker')
        self.Demo['list_d'] = (self.Demo['month'] >= self.Demo['bound']).astype(int)
        Listshare = self.Demo.pivot(index='month', columns='ticker', values='list_d')
        Listshare = Listshare.fillna(0)
        Listshare = Listshare[(Listshare.index >= self.begin) & (Listshare.index <= self.end)]
        return Listshare

    # 计算剔除未满交易日期股票
    def calc_YMshare(self,min_month=15,min_year=120):
        tradingM_data = self._csmAPI.api_TRD_Mnth(filename='TRD_Mnth.csv', col=['Stkcd', 'Trdmnt', 'Ndaytrd'])

        def check_tradeDate_num(data):
            data['month_tradeDate'] = data['Ndaytrd'].shift(1)
            data['year_tradeDate'] = data['Ndaytrd'].rolling(window=12, min_periods=6).sum().shift(1)
            return data

        tradingM_data = tradingM_data.groupby('ticker').apply(check_tradeDate_num)
        self.Demo = self.Demo.merge(tradingM_data, how='left',on=['ticker', 'month'])
        self.Demo['year_month'] = ((self.Demo['month_tradeDate'] >= min_month) & (self.Demo['year_tradeDate'] >= min_year)).astype(int)
        YMshare = self.Demo.pivot(index='month', columns='ticker', values='year_month')
        YMshare = YMshare.fillna(0)
        YMshare = YMshare[(YMshare.index >= self.begin) & (YMshare.index <= self.end)]
        return YMshare

    # 计算剔除ST股票
    def calc_STshare(self):
        print('正在计算STshare，需要花费较长时间……')
        tradingD_data = self._csmAPI.api_TRD_Dalyr(filename='TRD_Dalyr.csv', col=['Stkcd', 'Trddt', 'Trdsta'])
        # 不使用groupby提高运算效率
        tradingD_data = tradingD_data.pivot(index='tradeDate', columns='ticker', values='Trdsta')
        tradingD_data['month'] = tradingD_data.index.str[:7]
        tradingD_data = tradingD_data.groupby('month').apply(lambda x: pd.DataFrame(x.iloc[0, :]).T)
        tradingD_data = tradingD_data.melt(id_vars='month', var_name='ticker', value_name='Trdsta')
        tradingD_data['st'] = tradingD_data['Trdsta'].isin([1, 4, 7, 10, 13]).astype(int)
        self.Demo = self.Demo.merge(tradingD_data, how='left', on=['ticker', 'month'])
        STshare = self.Demo.pivot(index='month', columns='ticker', values='st')
        STshare = STshare.fillna(0)
        STshare = STshare[(STshare.index >= self.begin) & (STshare.index <= self.end)]
        return STshare

#################################################################################################################################
# 因子投资策略层
# 基础输入变量包括有exclude,ret,size和factor
# 可以提供月调仓的日频和月频收益时间序列，可以自行定义交易策略，构造不同数量的投资组合
# 策略思路：在每月换仓时获取不同投资组合的tickers组成由此计算组合收益，但是目前仅支持factor因子分组，可以优化使得自行加载外部数据和更广义的自定义策略
#################################################################################################################################

class Strategy():
    def __init__(self,exclude,size,factor):
        self._exclude=exclude
        self._size=size
        self._factor=factor
        #
        #self._groups_timelist=pd.DataFrame({})

    def fit(self,ret,type='day'):
        if type=='day':
            ret['month']=ret.index.str[:7]
        else:
            ret['month']=ret.index
        self._ret=ret

    def sort_group(self,df):
        # 传入数据：df.columns=['exclude','size','factor']，df.index=['ticker']
        # 传出数据：[[tickers1],[tickers2],……]
        df=df.sort_values('factor').dropna()
        group1=list(df.iloc[:int(len(df)*0.3),:].index)
        group2=list(df.iloc[int(len(df)*0.3):,:].index)
        group_list=[group1,group2]
        return group_list

    def _get_group(self,month):
        df=pd.concat([self._exclude.loc[month],self._factor.loc[month]],axis=1)
        df.columns=['exclude','factor']
        df=df[df['exclude']==1]
        if len(df)>11:
            #当期分组，若无则使用上期分组
            self._group_list=self.sort_group(df)
        # 保存分组时间序列
        # groups_timelist=pd.DataFrame(self._group_list,columns=range(len(self._group_list)),index=[month])
        # self._groups_timelist=pd.concat([self._groups_timelist,groups_timelist])

    def get_portfolio_ret(self,group_name):
        group_num=len(group_name)
        month_list = self._ret['month'].sort_values().unique().tolist()
        dfew = pd.DataFrame(index=self._ret.index,columns=list(range(group_num)))
        dfvw = pd.DataFrame(index=self._ret.index,columns=list(range(group_num)))
        for m,month in enumerate(month_list):
            print(month)
            final=self._ret[self._ret['month']==month]
            final=final.drop('month',axis=1)
            # 获取分组数据
            self._get_group(month)
            for i,stks in enumerate(self._group_list):
                s=self._size[stks].loc[month]
                p,pvw=final[stks],final[stks]
                for stk in stks:
                    pvw[stk]=pvw[stk]*s[stk]/s.sum()
                rew=pd.DataFrame(p.apply(lambda x:x.mean(),axis=1),columns=[i])
                rvw=pd.DataFrame(pvw.apply(lambda x:x.sum(),axis=1),columns=[i])
                dfew.update(rew)
                dfvw.update(rvw)
        dfew.columns = group_name
        dfvw.columns = group_name
        return dfew,dfvw

#################################################################################################################################
# 图片列表展示部分
# 基础输入变量可分为日度和月度，index表示时间，columns表示组合名称，valus表示组合收益*100
#################################################################################################################################

class Performance():
    def __init__(self,pew,pvw,path_ori,freq='day'):
        pew,pvw=pew.copy(),pvw.copy()
        pew.columns=pd.Series(pew.columns)+'_ew'
        pvw.columns=pd.Series(pvw.columns)+'_vw'
        portfolio=pd.concat([pew,pvw],axis=1).astype(float)
        # 设置调用接口对象导入FF因子数据
        self._csmAPI = Csmar_Api(path_ori=path_ori)
        self.freq=freq
        if freq=='day':
            self.ff5 = self._csmAPI.api_STK_MKT_FIVEFACDAY('STK_MKT_FIVEFACDAY.csv')
            data=portfolio.reset_index().melt(id_vars='tradeDate',var_name='ticker',value_name='exret')
            self.data = data.merge(self.ff5, how='left', on='tradeDate')
        else:
            self.ff5=self._csmAPI.api_STK_MKT_FIVEFACMONTH('STK_MKT_FIVEFACMONTH.csv')
            data = portfolio.reset_index().melt(id_vars='month', var_name='ticker', value_name='exret')
            self.data = data.merge(self.ff5, how='left', on='month')

    # Newey-West调整
    def _NWtest(self,a, lags=5):
        adj_a = pd.DataFrame(a)
        adj_a = adj_a.dropna()
        if len(adj_a) > 0:
            adj_a = adj_a.astype(float)
            adj_a = np.array(adj_a)
            model = sm.OLS(adj_a, [1] * len(adj_a)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
            return float(model.tvalues), float(model.pvalues)
        else:
            return [np.nan] * 2

    # 计算最大回撤
    def _maxdrawdown(self,return_list):
        return_list = list(return_list.fillna(0))
        ret_acc = []
        ret_sum = 1
        for i in range(len(return_list)):
            ret_sum = ret_sum * (return_list[i] + 1)
            ret_acc.append(ret_sum)
        ret_acc_max = np.maximum.accumulate(ret_acc)
        index_j = np.argmax(1 - ret_acc / ret_acc_max)
        index_i = np.argmax(ret_acc[:index_j])
        mdd = (ret_acc[index_i] - ret_acc[index_j]) / ret_acc[index_i]
        return mdd

    # 组合绩效
    def performance_valuation(self):
        data=self.data.copy()
        data['exret']=data['exret']/100
        # 计算单个组合
        def f(data):
            if self.freq=='day':
                ret_mean = data['exret'].mean() * 250
                std_mean = data['exret'].std() * np.sqrt(250)
            else:
                ret_mean = data['exret'].mean() * 12
                std_mean = data['exret'].std() * np.sqrt(12)
            ret_t = self._NWtest(data['exret'])[0]
            spr = ret_mean / std_mean
            # 最大回撤
            mdd = self._maxdrawdown(data['exret'])
            # alpha_CAMP
            CAPM = smf.ols('exret~mkt_rf', data=data).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            alpha_CAPM, t_CAPM = CAPM.params[0], CAPM.tvalues[0]
            # 特雷纳指标
            beta = CAPM.params[1]
            Treynor = ret_mean / beta
            # 信息比率
            resid_CAPM = CAPM.resid
            std_resid_CAMP = np.std(resid_CAPM)
            infomation_ratio = alpha_CAPM / std_resid_CAMP
            # alpha_FF3
            FF3 = smf.ols('exret~mkt_rf+smb+hml', data=data).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            alpha_FF3, t_FF3 = FF3.params[0], FF3.tvalues[0]
            # alpha_FF5
            FF5 = smf.ols('exret~mkt_rf+smb+hml+rmw+cma', data=data).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            alpha_FF5, t_FF5 = FF5.params[0], FF3.tvalues[0]
            res = pd.DataFrame(
                [ret_mean, ret_t, std_mean, beta, spr, mdd, Treynor, infomation_ratio, alpha_CAPM, t_CAPM, alpha_FF3,
                 t_FF3,
                 alpha_FF5, t_FF5]).T
            res.columns = ['ret_mean', 'ret_t', 'std_mean', 'beta', 'spr', 'mdd', 'Treynor', 'infomation_ratio',
                           'alpha_CAPM', 't_CAPM', 'alpha_FF3', 't_FF3', 'alpha_FF5', 't_FF5']
            return res
        res = data.groupby('ticker').apply(f)
        res.index = res.index.droplevel(1)
        return res

    def plot_exNetValue(self):
        data=self.data
        if self.freq=='day':
            data = data.pivot(index='tradeDate', columns='ticker', values='exret')
            data = data.expanding().apply(lambda x: (x/100 + 1).prod() - 1)
        else:
            data = data.pivot(index='month', columns='ticker', values='exret')
            data = data.expanding().apply(lambda x: (x/100 + 1).prod() - 1)
        fig, ax = plt.subplots(1, 1)
        data.plot(ax=ax, figsize=(9, 4))