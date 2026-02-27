# MALA 项目贡献指南

感谢您对 MALA 项目的兴趣！我们欢迎各种形式的贡献。

## 如何贡献

### 1. 报告问题
如果您发现了bug或有功能建议，请通过GitHub Issues页面提交。

### 2. 提交代码
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

### 3. 代码规范
- 使用 PEP 8 代码风格
- 添加详细的注释和文档字符串
- 确保代码通过现有测试

## 开发环境设置

```bash
# 克隆您的fork
git clone https://github.com/YOUR_USERNAME/MALA.git
cd MALA

# 创建虚拟环境
conda create -n mala python=3.9
conda activate mala

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果有

# 安装pre-commit hooks
pre-commit install
```

## 许可证

通过贡献代码，您同意您的贡献将在 MIT 许可证下发布。
