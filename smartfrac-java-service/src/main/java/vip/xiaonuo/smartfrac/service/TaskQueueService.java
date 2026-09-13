package vip.xiaonuo.smartfrac.service;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

/**
 * 基于 Redis 的任务队列服务
 * 使用 Redis List 作为任务队列：LPUSH 生产，BRPOP 消费
 */
@Service
@RequiredArgsConstructor
public class TaskQueueService {

    private final RedisTemplate<String, Object> redisTemplate;

    private static final String QUEUE_PREFIX = "smartfrac:queue:";
    private static final String TASK_KEY_PREFIX = "smartfrac:task:";

    /**
     * 提交任务到队列
     */
    public void submitTask(String taskType, Object taskPayload) {
        String queueKey = QUEUE_PREFIX + taskType;
        String taskId = String.valueOf(System.currentTimeMillis());

        // 存储任务详情
        redisTemplate.opsForValue().set(TASK_KEY_PREFIX + taskId, taskPayload, 1, TimeUnit.HOURS);

        // 推入队列
        redisTemplate.opsForList().leftPush(queueKey, taskId);
    }

    /**
     * 更新任务进度（Redis缓存）
     */
    public void updateProgress(String taskId, Integer progress, String status) {
        String key = TASK_KEY_PREFIX + taskId + ":progress";
        redisTemplate.opsForValue().set(key, status + ":" + progress, 30, TimeUnit.MINUTES);
    }

    /**
     * 获取任务进度
     */
    public String getProgress(String taskId) {
        String key = TASK_KEY_PREFIX + taskId + ":progress";
        Object val = redisTemplate.opsForValue().get(key);
        return val != null ? val.toString() : "unknown:0";
    }
}
