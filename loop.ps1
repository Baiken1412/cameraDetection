# ==========================================
# Ralph Playbook 循环执行器（Windows 版）
#
# 设计原则：
# - 每一轮都是"冷启动"
# - LLM 不保留任何跨轮上下文
# - 所有执行状态仅存在于磁盘文件中
# - Loop 本身不做决策，只负责反复调用
# ==========================================

param (
    # 执行模式：
    # plan  - 使用 PROMPT_plan.md（规划模式）
    # build - 使用 PROMPT_build.md（构建模式，默认）
    # fix   - 使用 PROMPT_fix.md（修复模式，省Token）
    [ValidateSet("plan", "build", "fix")]
    [string]$Mode = "build"
)

Write-Host "=========================================="
Write-Host "Ralph loop started (cameraDetection)"
Write-Host "Mode: $Mode"
Write-Host "State: IMPLEMENTATION_PLAN.md"
Write-Host "=========================================="

while ($true) {

    Write-Host ""
    Write-Host "------------------------------------------"
    Write-Host "New iteration: $(Get-Date)"
    Write-Host "------------------------------------------"

    if ($Mode -eq "plan") {
        Write-Host "[PLAN MODE]"
        Write-Host "Generating / rebuilding IMPLEMENTATION_PLAN.md"
        Get-Content PROMPT_plan.md | claude
    }
    elseif ($Mode -eq "fix") {
        Write-Host "[FIX MODE]"
        Write-Host "Locating bug -> minimal fix -> test -> commit"
        Get-Content PROMPT_fix.md | claude
    }
    else {
        Write-Host "[BUILD MODE]"
        Write-Host "Selecting one task -> implement -> test -> commit"
        Get-Content PROMPT_build.md | claude
    }

    Write-Host ""
    Write-Host "Iteration finished."
    Write-Host "Press Ctrl+C to exit, next iteration in 2s..."

    # 短暂休眠，防止过快循环
    Start-Sleep -Seconds 2
}
