package vip.xiaonuo.smartfrac.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import vip.xiaonuo.smartfrac.common.Result;

import java.util.HashMap;
import java.util.Map;

/**
 * 健康检查 Controller
 */
@RestController
@RequestMapping("/api/health")
public class HealthController {

    @GetMapping
    public Result<Map<String, Object>> health() {
        Map<String, Object> info = new HashMap<>();
        info.put("status", "UP");
        info.put("service", "smartfrac-java-service");
        info.put("version", "2.0.0-SNAPSHOT");
        return Result.success(info);
    }
}
