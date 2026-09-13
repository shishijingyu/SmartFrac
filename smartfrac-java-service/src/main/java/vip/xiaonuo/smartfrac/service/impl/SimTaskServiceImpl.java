package vip.xiaonuo.smartfrac.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import vip.xiaonuo.smartfrac.entity.SimTask;
import vip.xiaonuo.smartfrac.mapper.SimTaskMapper;
import vip.xiaonuo.smartfrac.service.SimTaskService;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.ThreadLocalRandom;

@Service
public class SimTaskServiceImpl extends ServiceImpl<SimTaskMapper, SimTask> implements SimTaskService {

    @Override
    public SimTask submitTask(Long caseId, String taskType, String inputFiles) {
        SimTask task = new SimTask();
        task.setTaskNo("TASK-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
                + "-" + ThreadLocalRandom.current().nextInt(1000, 9999));
        task.setCaseId(caseId);
        task.setTaskType(taskType);
        task.setStatus("pending");
        task.setProgress(0);
        task.setInputFiles(inputFiles);
        task.setCreateTime(LocalDateTime.now());
        this.save(task);
        return task;
    }

    @Override
    public void updateProgress(String taskNo, Integer progress, String status) {
        SimTask task = this.lambdaQuery()
                .eq(SimTask::getTaskNo, taskNo)
                .one();
        if (task != null) {
            task.setProgress(progress);
            task.setStatus(status);
            if ("running".equals(status) && task.getStartTime() == null) {
                task.setStartTime(LocalDateTime.now());
            }
            if ("completed".equals(status) || "failed".equals(status)) {
                task.setEndTime(LocalDateTime.now());
            }
            this.updateById(task);
        }
    }
}
