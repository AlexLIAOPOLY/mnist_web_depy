#!/usr/bin/env python
"""
修复markupsafe软件包中缺失的soft_unicode函数
这个问题发生在较新版本的markupsafe中，但旧版本的jinja2仍然依赖它
"""

import sys
import importlib

# 检查markupsafe是否已导入
if 'markupsafe' in sys.modules:
    markupsafe = sys.modules['markupsafe']
else:
    markupsafe = importlib.import_module('markupsafe')

# 检查soft_unicode是否已存在，若不存在则添加
if not hasattr(markupsafe, 'soft_unicode'):
    print("添加缺失的soft_unicode函数到markupsafe")
    
    def soft_unicode(s):
        """
        确保字符串是Unicode，与原来的函数行为相同
        参数：
            s: 任何对象
        返回：
            一个Unicode字符串
        """
        if isinstance(s, str):
            return s
        return str(s)
    
    # 添加函数到markupsafe命名空间
    markupsafe.soft_unicode = soft_unicode
    
    # 也添加到__all__列表，如果存在的话
    if hasattr(markupsafe, '__all__') and isinstance(markupsafe.__all__, list):
        markupsafe.__all__.append('soft_unicode')

print("MarkupSafe修复已应用") 