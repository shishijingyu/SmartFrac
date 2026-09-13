<template>
  <div>
    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>模型管理</span>
          <el-button type="primary" :icon="Upload">上传模型</el-button>
        </div>
      </template>
      <el-empty description="模型管理模块开发中，当前可在 results 目录查看已有模型" />
    </el-card>
  </div>
</template>

<script setup>
import { Upload } from '@element-plus/icons-vue'
</script>
