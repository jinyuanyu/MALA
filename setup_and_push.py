# -*- coding: utf-8 -*-
"""
================================================================================
MALA 项目初始化和推送脚本
================================================================================

此脚本用于初始化MALA项目并推送到GitHub。

使用前请确保：
1. 已安装 Git
2. 拥有 GitHub 账号
3. 已配置 Git 用户信息

Author: Jin Yuanyu
Email: jinyuanyu@example.com
"""

import os
import subprocess
import sys

# GitHub 配置
GITHUB_USERNAME = "jinyuanyu"
REPO_NAME = "MALA"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "YOUR_TOKEN_HERE")  # 请使用环境变量或替换为您的token

# 项目路径
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, cwd=None, capture_output=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            encoding='utf-8'
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def check_git_installed():
    """检查Git是否安装"""
    print("检查 Git 安装...")
    code, stdout, stderr = run_command("git --version")
    if code == 0:
        print(f"✓ Git 已安装: {stdout.strip()}")
        return True
    else:
        print("✗ Git 未安装")
        print("请从 https://git-scm.com 下载安装 Git")
        return False


def init_git_repo():
    """初始化Git仓库"""
    print("\n初始化 Git 仓库...")
    
    # 检查是否已是git仓库
    if os.path.exists(os.path.join(PROJECT_PATH, ".git")):
        print("✓ 已是 Git 仓库")
        return True
    
    code, stdout, stderr = run_command("git init", cwd=PROJECT_PATH)
    if code == 0:
        print("✓ Git 仓库初始化成功")
        return True
    else:
        print(f"✗ 初始化失败: {stderr}")
        return False


def configure_git():
    """配置Git用户信息"""
    print("\n配置 Git...")
    
    # 设置用户名和邮箱（需要用户配置）
    print("请配置 Git 用户信息（如果尚未配置）:")
    print("  git config --global user.name 'Your Name'")
    print("  git config --global user.email 'your@email.com'")


def create_initial_commit():
    """创建初始提交"""
    print("\n创建初始提交...")
    
    # 添加所有文件
    code, stdout, stderr = run_command("git add .", cwd=PROJECT_PATH)
    if code != 0:
        print(f"✗ 添加文件失败: {stderr}")
        return False
    
    # 检查是否有文件待提交
    code, stdout, stderr = run_command("git status --porcelain", cwd=PROJECT_PATH)
    if not stdout.strip():
        print("✗ 没有文件需要提交")
        return False
    
    # 创建提交
    commit_msg = "Initial commit: MALA - Masked Autoencoder for Remote Sensing Image Completion"
    code, stdout, stderr = run_command(f'git commit -m "{commit_msg}"', cwd=PROJECT_PATH)
    if code == 0:
        print("✓ 提交成功")
        return True
    else:
        print(f"✗ 提交失败: {stderr}")
        return False


def create_github_repo():
    """创建GitHub仓库"""
    print("\n创建 GitHub 仓库...")
    
    # 使用GitHub CLI或API创建仓库
    # 这里提供手动创建的方法
    
    print(f"""
请手动创建 GitHub 仓库：
1. 访问 https://github.com/new
2. Repository name 输入: {REPO_NAME}
3. 选择 Public 或 Private
4. 不要勾选 "Add a README file"
5. 点击 "Create repository"

创建完成后，请运行以下命令：
""")
    
    remote_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    
    print(f"git remote add origin {remote_url}")
    print(f"git branch -M main")
    print(f"git push -u origin main")
    
    return True


def push_to_github():
    """推送到GitHub"""
    print("\n推送到 GitHub...")
    
    remote_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    
    # 添加远程仓库
    code, stdout, stderr = run_command(f"git remote add origin {remote_url}", cwd=PROJECT_PATH)
    if code != 0 and "already exists" not in stderr.lower():
        print(f"⚠ 添加远程仓库: {stderr}")
    
    # 重命名分支
    code, stdout, stderr = run_command("git branch -M main", cwd=PROJECT_PATH)
    
    # 推送
    code, stdout, stderr = run_command("git push -u origin main", cwd=PROJECT_PATH)
    
    if code == 0:
        print("✓ 推送成功!")
        print(f"仓库地址: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
        return True
    else:
        print(f"✗ 推送失败: {stderr}")
        print("\n请手动运行以下命令:")
        print(f"  git remote add origin https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git")
        print(f"  git push -u origin main")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("MALA 项目初始化和推送脚本")
    print("=" * 60)
    
    # 检查Git
    if not check_git_installed():
        print("\n请先安装 Git: https://git-scm.com")
        return
    
    # 配置Git
    configure_git()
    
    # 初始化仓库
    if not init_git_repo():
        return
    
    # 创建提交
    if not create_initial_commit():
        return
    
    # 推送到GitHub
    push_to_github()
    
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
