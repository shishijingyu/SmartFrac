<template>
  <div>
    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>算例管理</span>
          <el-button type="primary" :icon="Plus">新建算例</el-button>
        </div>
      </template>
      <el-empty description="暂无算例，待接入第一阶段 yaml 配置" />
    </el-card>
  </div>
</template>

<script setup>
import { Plus } from '@element-plus/icons-vue'
</script>
