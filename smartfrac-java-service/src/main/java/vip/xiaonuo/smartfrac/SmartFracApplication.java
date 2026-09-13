package vip.xiaonuo.smartfrac;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * SmartFrac Java 业务服务层启动类
 *
 * @author SmartFrac
 */
@SpringBootApplication
@MapperScan("vip.xiaonuo.smartfrac.mapper")
public class SmartFracApplication {

    public static void main(String[] args) {
        SpringApplication.run(SmartFracApplication.class, args);
        System.out.println("""
                ============================================================
                  SmartFrac Java Service 启动成功!
                  多尺度-不确定性耦合AI断裂智能仿真系统
                  后端地址: http://localhost:48091
                ============================================================
                """);
    }
}
