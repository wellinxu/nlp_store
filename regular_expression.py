"""
正则示例
"""
import re


# 测试文本
text = "这是一段测试文本,test text for testing，这里面包含了数字0-9(\d)，字母a-z，中文，邮编210000等等"
print("测试文本：", text)

# 匹配6个数字
regress = r"\d{6}"
result = re.findall(regress, text)
print("\n匹配6个数字")
print(result)    # ['210000']

# 匹配2-7个中文
regress = r"[\u4e00-\u9fa5]{2,7}"
result = re.findall(regress, text)
print("\n匹配2-7个中文")
print(result)    # ['这是一段测试文', '这里面包含了数', '字母', '中文', '邮编', '等等']

# 匹配几个中文
regress = r"这是[\u4e00-\u9fa5]+"
result = re.findall(regress, text)
print("\n匹配几个中文")
print(result)    # ['这是一段测试文本']

# 匹配字符按串'中文'或者'文'
regress = r"中文|文"
regress = r"中?文"
result = re.findall(regress, text)
print("\n匹配字符按串'中文'或者'文'")
print(result)    # ['文', '中文']

# 匹配中文+字母或者字母
regress = r"[\u4e00-\u9fa5]*[a-z]+"
result = re.findall(regress, text)
print("\n匹配中文+字母或者字母")
print(result)    # ['test', 'text', 'for', 'testing', 'd', '字母a', 'z']

# 匹配9或a
regress = r"[9a]"
regress = r"9|a"
result = re.findall(regress, text)
print("\n匹配9或a")
print(result)    # ['9', 'a']

# 匹配非中文
regress = r"[^\u4e00-\u9fa5]+"
result = re.findall(regress, text)
print("\n匹配非中文")
print(result)    # [',test text for testing，', '0-9(\\d)，', 'a-z，', '，', '210000']

# 匹配字符串'\d'
regress = r"\\d"
result = re.findall(regress, text)
print("\n匹配字符串'\d'")
print(result)    # ['\\d']

# 匹配开头
regress = r"^这"
result = re.search(regress, text)
print("\n匹配开头")
print(result)    # <_sre.SRE_Match object; span=(0, 1), match='这'>

# 匹配结尾
regress = r"等$"
result = re.search(regress, text)
print("\n匹配结尾")
print(result)    # <_sre.SRE_Match object; span=(65, 66), match='等'>

# 匹配单词边界
regress = r"\btest\b"
result = re.finditer(regress, text)
print("\n匹配单词边界")
print([i for i in result])    # [<_sre.SRE_Match object; span=(9, 13), match='test'>]
regress = r"\btest"
result = re.finditer(regress, text)
# [<_sre.SRE_Match object; span=(9, 13), match='test'>, <_sre.SRE_Match object; span=(23, 27), match='test'>]
print([i for i in result])
regress = r"test\b"
result = re.finditer(regress, text)
print([i for i in result])    # [<_sre.SRE_Match object; span=(9, 13), match='test'>]

# 匹配‘这是...文’
# 贪婪匹配,匹配这个正则能够匹配到的最长的文本，大多数都是默认贪婪的
regress = r"这是.*文"
result = re.findall(regress, text)
print("\n匹配‘这是...文’")
print(result)    # ['这是一段测试文本,test text for testing，这里面包含了数字0-9(\\d)，字母a-z，中文']

# 匹配‘这是...文’，非贪婪
# 非贪婪匹配，匹配这个正则能够匹配到的最短文本，在可变长匹配符号后面加?
regress = r"这是.*?文"
result = re.findall(regress, text)
print("\n匹配‘这是...文’，非贪婪")
print(result)    # ['这是一段测试文']


# #分组与捕获
# 匹配'这是一段测试文本'，获取组
regress = r"这是一段((测试)(文本))"
result = re.search(regress, text)
print("\n匹配'这是一段测试文本'，获取组")
# <_sre.SRE_Match object; span=(0, 8), match='这是一段测试文本'>
print(result)
print("第三个分组：")
print(result.group(3))    # 文本
new_text = re.sub(regress, r"【\2】", text)
print("将匹配的文本替换成第二个组里面的内容：")
# 【测试】,test text for testing，这里面包含了数字0-9(\d)，字母a-z，中文，邮编210000等等
print(new_text)

# 匹配‘测试文’或‘文’
regress = r"(测试)?文"
result = re.finditer(regress, text)
print("\n匹配‘测试文’或‘文’")
# [<_sre.SRE_Match object; span=(4, 7), match='测试文'>, <_sre.SRE_Match object; span=(54, 55), match='文'>]
print([i for i in result])

# 匹配‘这里’或‘这是’
regress = r"这(里|是)"
result = re.finditer(regress, text)
print("\n匹配‘这里’或‘这是’")
# [<_sre.SRE_Match object; span=(0, 2), match='这是'>, <_sre.SRE_Match object; span=(31, 33), match='这里'>]
print([i for i in result])


# 回溯引用
# \n表示引用了第n个分组的结果，可以想象为变量
regress = r"t(es).*t\1"
result = re.search(regress, text)
print("\n回溯引用")
print(result)    # <_sre.SRE_Match object; span=(9, 26), match='test text for tes'>


# #前后查找

# 匹配0，且0后面得是3个数字
# 这个位置的(?=exp)表示匹配字符的后面得满足exp
regress = r"0(?=\d{3})"
result = re.search(regress, text)
print("\n匹配0，且0后面得是3个数字")
print(result)    # <_sre.SRE_Match object; span=(60, 61), match='0'>

# 匹配0，且0前面是中文
# 这个位置的(?<=exp)表示匹配字符的前面得满足exp
regress = r"(?<=[\u4e00-\u9fa5])0"
result = re.search(regress, text)
print("\n匹配0，且0前面是中文")
print(result)    # <_sre.SRE_Match object; span=(39, 40), match='0'>

text = "这是一段测试文本，这里面包含了数字0-9，字母a-z，中文，邮编210000等等"
# 匹配字母，且后面不能是-
regress = r"[a-z](?!\-)"
# 这个位置的(?!exp)表示匹配字符的后面不能满足exp
result = re.findall(regress, text)
print("\n匹配字母，且后面不能是-")
print(result)    # ['z']

# 匹配字母，且前面不能是-
# 这个位置的(?<!exp)表示匹配字符的前面不能满足exp
regress = r"(?<!\-)[a-z]"
result = re.findall(regress, text)
print("\n匹配字母，且前面不能是-")
print(result)    # ['a']

# 匹配3个数字，要以10开头
# 这个位置的(?=exp)表示匹配字符的开始得满足exp
regress = r"(?=10)\d{3}"
result = re.findall(regress, text)
print("\n匹配3个数字，要以10开头")
print(result)    # ['100']

# 匹配3个数字，不要以2或1开头
# 这个位置的(?!exp)表示匹配字符的开始不能满足exp
regress = r"(?!2|1)\d{3}"
result = re.findall(regress, text)
print("\n匹配3个数字，不要以2或1开头")
print(result)    # ['000']

# 匹配3个数字，要以00结尾
# //这个位置的(?<=exp)表示匹配字符的结尾得满足exp
regress = r"\d{3}(?<=00)"
result = re.findall(regress, text)
print("\n匹配3个数字，要以00结尾")
print(result)    # ['100']

# # 匹配3个数字，不要以10或100结尾
# # 这个位置的(?<!exp)表示匹配字符的结尾不能满足exp
# # python不支持这种正则
# regress = r"\d{3}(?<!10|100)"
# result = re.findall(regress, text)
# print("\n匹配3个数字，不要以10或100结尾")
# print(result)    # ['000']


# #条件匹配

# 匹配字符‘文’，如果‘文’子前面有测试，则多匹配任意一个字符
# (?(n)exp1|exp2)表示，如果第n个分组被匹配到，则匹配exp1,否者匹配exp2，此语法中n不需要转义
regress = r"(测试)?文(?(1).|.{2})"
result = re.search(regress, text)
print("\n匹配字符‘文’，如果‘文’子前面有测试，则多匹配任意一个字符")
print(result)    # <_sre.SRE_Match object; span=(4, 8), match='测试文本'>


