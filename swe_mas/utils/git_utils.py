"""Git工具函数"""

import os
import shutil
import subprocess
import time
import hashlib
from pathlib import Path

from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


def configure_git_ssh() -> None:
    """配置Git使用SSH
    
    设置环境变量，强制Git使用SSH而不是HTTPS
    优先使用环境变量指定的密钥，否则尝试常见的SSH密钥路径
    """
    # 检查环境变量指定的SSH密钥路径
    ssh_key_path = os.getenv("GIT_SSH_KEY_PATH")
    
    # 如果没有指定，尝试常见的SSH密钥路径
    if not ssh_key_path:
        home = Path.home()
        common_keys = [
            home / ".ssh" / "id_ed25519",
            home / ".ssh" / "id_rsa",
            home / ".ssh" / "id_ecdsa",
        ]
        for key_path in common_keys:
            if key_path.exists():
                ssh_key_path = str(key_path)
                break
    
    # 设置Git SSH命令
    if ssh_key_path and Path(ssh_key_path).exists():
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        logger.info(f"Git配置为使用SSH密钥: {ssh_key_path}")
    else:
        # 使用默认SSH配置（依赖ssh-agent）
        os.environ["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        logger.info("Git配置为使用默认SSH（通过ssh-agent）")


def convert_https_to_ssh(url: str) -> str:
    """将HTTPS Git URL转换为SSH格式
    
    Args:
        url: Git仓库URL (HTTPS或SSH格式)
        
    Returns:
        SSH格式的URL
        
    Examples:
        https://github.com/user/repo.git -> git@github.com:user/repo.git
        git@github.com:user/repo.git -> git@github.com:user/repo.git (不变)
    """
    if url.startswith("https://github.com/"):
        # GitHub HTTPS -> SSH
        path = url.replace("https://github.com/", "")
        return f"git@github.com:{path}"
    elif url.startswith("https://gitlab.com/"):
        # GitLab HTTPS -> SSH
        path = url.replace("https://gitlab.com/", "")
        return f"git@gitlab.com:{path}"
    elif url.startswith("https://"):
        # 通用转换
        # https://domain.com/path/repo.git -> git@domain.com:path/repo.git
        url_without_protocol = url.replace("https://", "")
        parts = url_without_protocol.split("/", 1)
        if len(parts) == 2:
            domain, path = parts
            return f"git@{domain}:{path}"
    
    # 已经是SSH格式或无法识别的格式，保持不变
    return url


def clone_repo_ssh(repo_url: str, target_dir: str, use_ssh: bool = True, max_retries: int = 3) -> bool:
    """克隆Git仓库（支持SSH，带重试机制）
    
    Args:
        repo_url: 仓库URL
        target_dir: 目标目录
        use_ssh: 是否强制使用SSH（默认True）
        max_retries: 最大重试次数（默认3次）
        
    Returns:
        是否成功
    """
    # 配置SSH
    if use_ssh:
        configure_git_ssh()
        repo_url = convert_https_to_ssh(repo_url)
        logger.info(f"使用SSH克隆: {repo_url}")
    
    # 设置环境变量，增加Git缓冲区大小和超时
    env = os.environ.copy()
    env["GIT_HTTP_BUFFER"] = "524288000"  # 500MB 缓冲区
    env["GIT_HTTP_LOW_SPEED_LIMIT"] = "1000"  # 1KB/s 最低速度
    env["GIT_HTTP_LOW_SPEED_TIME"] = "600"  # 600秒超时
    
    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"克隆尝试 {attempt}/{max_retries}: {repo_url}")
            
            result = subprocess.run(
                ["git", "clone", repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=900,  # 15分钟超时（大型仓库需要更长时间）
                env=env,
            )
            
            if result.returncode == 0:
                logger.info(f"仓库克隆成功: {target_dir}")
                return True
            else:
                error_msg = result.stderr.strip()
                logger.warning(f"克隆尝试 {attempt} 失败: {error_msg}")
                
                # 清理失败的克隆目录
                target_path = Path(target_dir)
                if target_path.exists():
                    try:
                        shutil.rmtree(target_path)
                    except Exception:
                        pass
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries:
                    wait_time = attempt * 5  # 递增等待时间：5秒、10秒、15秒
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"仓库克隆失败（已重试 {max_retries} 次）: {error_msg}")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.warning(f"克隆尝试 {attempt} 超时")
            # 清理失败的克隆目录
            import shutil
            target_path = Path(target_dir)
            if target_path.exists():
                try:
                    shutil.rmtree(target_path)
                except Exception:
                    pass
            
            if attempt < max_retries:
                wait_time = attempt * 5
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"仓库克隆超时（已重试 {max_retries} 次）")
                return False
                
        except Exception as e:
            logger.error(f"克隆异常: {str(e)}")
            if attempt < max_retries:
                wait_time = attempt * 5
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return False
    
    return False


def ensure_git_repo(directory: str) -> bool:
    """确保目录是git仓库，如果不是则初始化
    
    Args:
        directory: 目录路径
        
    Returns:
        是否成功（True=已有或成功初始化，False=失败）
    """
    git_dir = Path(directory) / ".git"
    
    if git_dir.exists():
        logger.debug(f"Git仓库已存在: {directory}")
        return True
    
    try:
        logger.info(f"初始化Git仓库: {directory}")
        result = subprocess.run(
            ["git", "init"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            logger.info("Git仓库初始化成功")
            
            # 添加初始提交（方便git diff）
            subprocess.run(["git", "add", "."], cwd=directory, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit", "--allow-empty"],
                cwd=directory,
                capture_output=True,
            )
            
            return True
        else:
            logger.error(f"Git初始化失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Git初始化异常: {str(e)}")
        return False


def check_git_available() -> bool:
    """检查git命令是否可用
    
    Returns:
        git是否可用
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_ssh_available() -> bool:
    """检查SSH是否可用
    
    Returns:
        SSH是否可用
    """
    try:
        result = subprocess.run(
            ["ssh", "-V"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # SSH的版本信息在stderr中
        return result.returncode == 0 or "OpenSSH" in result.stderr
    except Exception:
        return False


def setup_git_config(use_ssh: bool = True) -> None:
    """设置Git全局配置
    
    Args:
        use_ssh: 是否配置为优先使用SSH
    """
    if use_ssh:
        configure_git_ssh()
        
        # 设置Git配置，优先使用SSH
        try:
            # 替换HTTPS为SSH（全局配置）
            subprocess.run(
                ["git", "config", "--global", "url.git@github.com:.insteadOf", "https://github.com/"],
                capture_output=True,
                timeout=5,
            )
            logger.info("Git全局配置已设置为使用SSH")
        except Exception as e:
            logger.warning(f"设置Git配置失败（非关键）: {str(e)}")


def get_env_snapshot(cwd: str | None = None) -> str:
    """获取当前 repo 的快照：HEAD + dirty + diff_hash；失败返回空字符串"""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if head.returncode != 0:
            return ""
        head_hash = head.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = "dirty" if status.stdout.strip() else "clean"

        diff = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        diff_text = diff.stdout or ""
        diff_hash = hashlib.sha1(diff_text[:200_000].encode("utf-8", errors="ignore")).hexdigest()[:12]

        return f"{head_hash}:{dirty}:{diff_hash}"
    except Exception:
        return ""
