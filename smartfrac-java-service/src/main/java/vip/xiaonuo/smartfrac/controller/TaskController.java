package vip.xiaonuo.smartfrac.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import vip.xiaonuo.smartfrac.common.Result;
import vip.xiaonuo.smartfrac.entity.SimTask;
import vip.xiaonuo.smartfrac.service.SimTaskService;
import vip.xiaonuo.smartfrac.service.TaskQueueService;

import java.util.List;
import java.util.Map;

/**
 * 仿真任务 Controller
 */
@RestController
@RequestMapping("/api/tasks")
public class TaskController {

    @Autowired
    private SimTaskService taskService;

    @Autowired
    private TaskQueueService taskQueueService;

    @GetMapping
    public Result<List<SimTask>> list(@RequestParam(required = false) String status) {
        LambdaQueryWrapper<SimTask> wrapper = new LambdaQueryWrapper<>();
        if (status != null && !status.isEmpty()) {
            wrapper.eq(SimTask::getStatus, status);
        }
        wrapper.orderByDesc(SimTask::getCreateTime);
        return Result.success(taskService.list(wrapper));
    }

    @GetMapping("/{id}")
    public Result<SimTask> getById(@PathVariable Long id) {
        return Result.success(taskService.getById(id));
    }

    @PostMapping
    public Result<SimTask> submit(@RequestBody Map<String, Object> body) {
        Long caseId = Long.valueOf(body.get("caseId").toString());
        String taskType = (String) body.get("taskType");
        String inputFiles = (String) body.get("inputFiles");

        SimTask task = taskService.submitTask(caseId, taskType, inputFiles);

        // 推入 Redis 任务队列，等待 Python 内核消费
        taskQueueService.submitTask(taskType, Map.of(
            "taskNo", task.getTaskNo(),
            "caseId", caseId,
            "type", taskType,
            "inputFiles", inputFiles != null ? inputFiles : ""
        ));

        return Result.success(task);
    }

    @PostMapping("/{id}/cancel")
    public Result<Void> cancel(@PathVariable Long id) {
        SimTask task = taskService.getById(id);
        if (task != null) {
            task.setStatus("cancelled");
            taskService.updateById(task);
        }
        return Result.success();
    }
}
