# 定义参数
dcu_hours = 1_488_137.26  # 使用的DCU hours
tdp_kw = 300 / 1000  # 将TDP转换为千瓦
carbon_intensity = 0.475  # 每千瓦时的碳排放量（kg/kWh），根据IEA的全球平均值

# 计算碳排放量
carbon_emission = dcu_hours * tdp_kw * carbon_intensity
print(carbon_emission)
