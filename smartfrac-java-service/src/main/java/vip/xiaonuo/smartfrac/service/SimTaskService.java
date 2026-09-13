package vip.xiaonuo.smartfrac.service;

import com.baomidou.mybatisplus.extension.service.IService;
import vip.xiaonuo.smartfrac.entity.SimTask;

public interface SimTaskService extends IService<SimTask> {

    /**
     * 提交仿真任务
     */
    SimTask submitTask(Long caseId, String taskType, String inputFiles);

    /**
     * 更新任务进度
     */
    void updateProgress(String taskNo, Integer progress, String status);
}
